'''
Draw a batch from cifar10
mask out from images in batch a certain fraction of patches randomly
return (i) masked images (ii) resnet features of masked images (iii) mask indices 
'''
import os
import sys
# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from classifier.utils import get_config
from classifier.data import kspaceCIFAR10
from classifier.masks.patch_mask import PatchMask
from classifier.classification_modules.pretrained_resnet import PreTrainedResNet

import torchvision.utils as vutils
from PIL import Image
    

def softmax(logits):
    max_ = np.max(logits, 1, keepdims=True)
    a0 = logits - max_
    e0 = np.exp(a0)
    z0 = np.sum(e0, 1, keepdims=True)
    softmax = e0 / z0
    return softmax

def cross_entropy(logits, labels):
    logits = logits.astype(np.float64)
    prob = softmax(logits)
    return -np.log(prob[np.arange(len(logits)), labels]), prob

def sigmoid(logits):
    logits = logits.astype(np.float64)
    z = np.where(logits >= 0, np.exp(-logits), np.exp(logits))
    return np.where(logits >= 0, 1 /(1 + z) , z / (1 + z))

def binary_cross_entropy(logits, labels):
    logits = logits.astype(np.float64)
    # follow pytorch implement
    if not (labels.shape == logits.shape):
        # raise ValueError("Target size ({}) must be the same as input size ({})".format(labels.shape, input.shape))
        raise ValueError(f"Target size ({labels.shape}) must be the same as input size ({logits.shape})")
    max_val = np.maximum((-logits), 0)
    loss = logits - logits * labels + max_val + np.log(np.exp(-max_val) + np.exp(-logits - max_val))
    p = sigmoid(logits)

    if np.any(loss == np.inf):
        loss[loss == np.inf] = np.maximum(logits[loss==np.inf],
                -logits[loss==np.inf])
    return loss, p


def data_load_cifar(args): #data_type=args.data_type, random_seed=args.random_seed, cost_from_file=args.cost_from_file
    # Set up data

    train_dataset = kspaceCIFAR10(args, mode="train")
    val_dataset = kspaceCIFAR10(args, mode="val")
    test_dataset = kspaceCIFAR10(args, mode="test")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.n_envs,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.n_envs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.n_envs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )

    cost = None # Just maintaining original code structure
    return (DataTempCifar(args, train_loader, len(train_dataset), True), \
            DataTempCifar(args, val_loader, len(val_dataset), True,
                deterministic_masks=True, mask_seed_offset=100000), \
            DataTempCifar(args, val_loader, len(val_dataset), True,
                deterministic_masks=True, mask_seed_offset=200000), \
            DataTempCifar(args, test_loader, len(test_dataset), True,
                deterministic_masks=True, mask_seed_offset=300000)), cost


# need init : n_features, n_actions, n_data, 
# need next_batches to return :  (images, mask_indices, labels, exist)
class DataTempCifar:
    def __init__(
        self,
        args,
        loader,
        n_data,
        iter_forever=True,
        deterministic_masks=False,
        mask_seed_offset=0,
    ):
        self.n_classes = 10 # cifar10
        self.loader = loader
        self.n_data = n_data
        self.args = args
        self.iter = iter_forever
        self._it = iter(self.loader) 

        self.patch_size = args.patch_size if hasattr(args, 'patch_size') else 4
        self.initial_visible_count = args.initial_visible_count
        self.deterministic_masks = deterministic_masks
        self.mask_seed_offset = mask_seed_offset
        B, C, H, W = self.args.n_envs, 3, 32, 32 # fixed for cifar10
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
        total_patches = num_patches_h * num_patches_w

        self.n_features = total_patches  # 64 for cifar10 with patch_size 4
        self.n_actions = self.n_features + 1  # +1 for terminal action

        self.index = 0  # to keep track of current index in data
        
    def _create_mask_indices(self, batch_size, start_index=0):
        B, C, H, W = batch_size, 3, 32, 32 # fixed for cifar10
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
        total_patches = num_patches_h * num_patches_w
        mask = torch.zeros(B, total_patches, dtype=torch.uint8)
        for b in range(B):
            if self.deterministic_masks:
                sample_index = (start_index + b) % self.n_data
                rng = np.random.default_rng(
                    self.args.random_seed + self.mask_seed_offset + sample_index
                )
                patch_indices = torch.from_numpy(
                    rng.permutation(total_patches)[:self.initial_visible_count]
                )
            else:
                patch_indices = torch.randperm(total_patches)[:self.initial_visible_count]
            mask[b, patch_indices] = 1
        # shaan : line below
        flattened_mask_indices = mask
        return flattened_mask_indices.cpu().numpy()
    

    def reset(self):
        self._it = iter(self.loader)
        self.index = 0


    def next_batch(self, batch_size):
        # Pull batches until we accumulate exactly batch_size samples
        images_list, labels_list = [], []
        got = 0
        while got < batch_size:
            try:
                batch = next(self._it)
            except StopIteration:
                if not self.iter:
                    return None
                self.reset()
                batch = next(self._it)

            imgs = batch["recon_esc"]               # torch [B,3,32,32]
            labs = batch["label"]["cifar10"].squeeze(-1)  # torch [B]

            # Move to CPU and numpy
            imgs = imgs.detach().cpu().numpy().astype(np.float32)
            labs = labs.detach().cpu().numpy().astype(np.int64)

            images_list.append(imgs)
            labels_list.append(labs)
            got += imgs.shape[0]

        images = np.concatenate(images_list, axis=0)[:batch_size]
        labels = np.concatenate(labels_list, axis=0)[:batch_size]

        mask_indices = self._create_mask_indices(
            batch_size, start_index=self.index
        ).astype(np.uint8)  # np uint8 (B,64)
        assert mask_indices.shape == (batch_size, self.n_features)
        exist = np.ones((batch_size, self.n_features), dtype=np.uint8)

        # advance index similarly to original
        self.index = (self.index + batch_size) % self.n_data if self.iter else min(self.index + batch_size, self.n_data)

        return images, mask_indices, labels, exist




class CifarEnv(object):
    def __init__(self, args, n_envs, data, classify_fn):

        self.n_features = data.n_features # must be 64
        self.n_classes = data.n_classes  # must be 10 as cifar10
        self.n_envs = n_envs # ~128
        self.r_cost = np.array(args.r_cost) 
        self.classify = classify_fn
        # self.input_size = (self.n_features, self.n_features + 1) # comback
        self.data = data
        self.flag = False

        # NOTE No need for below block as cifar10 has 10 classes balanced (assumes no binary classification)
        # self.pos_weight = 1 
        # if self.n_classes == 2:
        #     n_neg = len(np.where(data_labels == 0)[0])
        #     n_pos = len(np.where(data.labels == 1))
        #     self.pos_weight = n_neg *1. / n_pos

        # self.n_data = data.features.shape[0] # TODO data.features? need to check what this is exactly 60,000 for cifar10 train set?
        self.n_data = data.n_data
        self.n_epoch = 0
        assert self.n_data >= self.n_envs

        # NOTE: org data.next_batch output was (features, labels, exist) now changed to (images, mask_indices, labels, exist)
        batch = data.next_batch(self.n_envs) # ~128 in num and output is (images, mask_indices, labels, exist)
        self.batch_inputs_images = batch[0] 
        self.batch_inputs_mask_indices = batch[1] # must be of shape (n_envs, n_features) or (n_envs, 64)
        self.batch_labels = batch[2]
        self.batch_exist = batch[3] if batch[3] is not None else np.ones((self.n_envs, self.n_features), dtype=np.uint8).astype(np.uint8)
        
        # self.n_actions = self.batch_exist.shape[1] + 1 
        self.n_actions = self.n_features + 1  # +1 for terminal action

        # TODO check below modification of batch_obs
        # this is was original batch_obs now split into batch_obs_images and batch_obs_features
        # self.batch_obs = np.zeros((self.n_envs, self.n_features, self.n_features + 1), dtype=np.float32).astype(np.float32)
        # below is new batch_obs split where each 0th dim index of batch_obs_image corresponds to 0th dim index of batch_obs_features
        self.batch_obs_images = np.zeros((self.n_envs, 3, 32, 32), dtype=np.float32) # should be 3, 32, 32 for cifar10 images
        self.batch_obs_features = np.zeros((self.n_envs, self.n_features+1), dtype=np.uint8) # +1 for terminal action
        
        self.batch_obs_images[...] = self.batch_inputs_images
        self.batch_obs_features[:, :self.n_features] = self.batch_inputs_mask_indices  # if mask is 64-d
        self.batch_obs_features[:, self.n_features] = 0  # STOP bit off

        self.batch_acquired = np.zeros((self.n_envs, self.n_features)).astype(np.uint8) # self.batch_acquired is same as batch_obs_features except for terminal action. 
        self.batch_acquired[...] = self.batch_inputs_mask_indices
        # Have kept both batch_acquired and batch_obs_features for now to minimize changes, could have kept one one and modified code accordingly
        
        self.batch_returns = np.zeros(self.n_envs, dtype=np.float32)

        self.rewards = np.zeros(self.n_envs, dtype=np.float32)

        self.q_a = np.zeros((self.n_envs, self.n_actions),
                dtype=np.float32)
        

        #  NOTE this comment : "TODO : gpu.. .cuda()" was in original code, dont know if it needs to be acted upon. leavin it for now
        # shaan : These tensors share memory with numpy arrays. Updating numpy updates torch and vice versa (with caveats) - may only be for cpu
        self.var_inputs_images = torch.from_numpy(self.batch_inputs_images) 
        self.var_inputs_mask_indices = torch.from_numpy(self.batch_inputs_mask_indices) 
        self.var_labels = torch.from_numpy(self.batch_labels)
        self.var_exist = torch.from_numpy(self.batch_exist)
        self.var_acquired = torch.from_numpy(self.batch_acquired)
        self.var_obs_images = torch.from_numpy(self.batch_obs_images)
        self.var_obs_features = torch.from_numpy(self.batch_obs_features)
        self.var_rewards = torch.from_numpy(self.rewards)

        # NOTE commented out below block as n_actions = n_features + 1 always for cifar10
        # if self.n_actions != self.n_features + 1:
        #     self.batch_acquired_aux = np.zeros_like(self.batch_exist)
        #     self.var_acquired_aux = torch.from_numpy(self.batch_acquired_aux)
        

    def calc_reward(self, p_y_logit, labels): # NOTE is same as in original code
        if self.n_classes > 2:
            crss_ent, prob = cross_entropy(p_y_logit, labels)
        else:
            crss_ent, prob = binary_cross_entropy(p_y_logit.reshape(-1),
                labels)
        loglikelihood = -crss_ent
        assert loglikelihood.shape[0] == prob.shape[0]
        return loglikelihood, prob

    def reset(self, first=True, reset_data=None): # shaan : note that in main reset_data is always None
        if reset_data is not None: # So, I think can ignore this block
            self.data = reset_data
            self.n_data = self.data.shape[0] # or self.data.data.shape[0]?
        if first:
            self.data.reset()
            self.n_epoch = 0
        batch = self.data.next_batch(self.n_envs)
        self.batch_inputs_images = batch[0] 
        self.batch_inputs_mask_indices = batch[1]
        self.batch_labels = batch[2]
        self.batch_exist = batch[3] if batch[3] is not None else np.ones((self.n_envs, self.n_features), dtype=np.uint8).astype(np.uint8)

        self.batch_acquired = np.zeros((self.n_envs, self.n_features)).astype(np.uint8)
        self.batch_acquired[...] = self.batch_inputs_mask_indices

        self.batch_obs_images = np.zeros((self.n_envs, 3, 32, 32), dtype=np.float32) # should be 3, 32, 32 for cifar10 images
        self.batch_obs_features = np.zeros((self.n_envs, self.n_features+1), dtype=np.uint8) # +1 for terminal action

        self.batch_obs_images[...] = self.batch_inputs_images
        self.batch_obs_features[:, :self.n_features] = self.batch_inputs_mask_indices  # if mask is 64-d
        self.batch_obs_features[:, self.n_features] = 0  # STOP bit off

        self.batch_returns = np.zeros(self.n_envs, dtype=np.float32)
        self.rewards = np.zeros(self.n_envs, dtype=np.float32)

        self.q_a = np.zeros((self.n_envs, self.n_actions),
                dtype=np.float32)
    
        self.var_inputs_images = torch.from_numpy(self.batch_inputs_images) 
        self.var_inputs_mask_indices = torch.from_numpy(self.batch_inputs_mask_indices) 
        self.var_labels = torch.from_numpy(self.batch_labels)
        self.var_exist = torch.from_numpy(self.batch_exist)
        self.var_acquired = torch.from_numpy(self.batch_acquired)
        self.var_obs_images = torch.from_numpy(self.batch_obs_images)
        self.var_obs_features = torch.from_numpy(self.batch_obs_features)
        self.var_rewards = torch.from_numpy(self.rewards)


    def get_current_batch_with_random_features(self):
        # below line was original 
        # sampled = np.random.binomial(1, 0.5, self.batch_exist.shape) * self.batch_exist.astype(int)
        # below line replaces that to match self.batch_obs_features
        sampled = (np.random.binomial(1, 0.2, size=(self.n_envs, self.n_features)).astype(int))
        sampled = sampled * self.batch_exist.astype(int) # not needed really as batch_exist is all ones for cifar10
 
        # no need for this line
        # batch_obs = np.zeros((self.n_envs, self.n_features, self.n_features + 1), dtype=np.float32).astype(np.float32) 

        self.n_actions == self.n_features + 1
        if self.n_actions == self.n_features + 1:
            length = np.sum(sampled, axis=1).astype(np.int64)
            # x, y = np.where(sampled) no need
        else:
            # NOTE this block will not be executed for cifar10 as n_actions = n_features + 1 always
            pass

        # NOTE: sampled behaves as self.batch_obs_features except for terminal action
        # NOTE: most likely length wont be required
        return torch.from_numpy(self.batch_inputs_images), torch.from_numpy(sampled), self.var_labels, torch.from_numpy(length)
    
    def get_current_batch_with_random_features_2(self):
        if self.flag == False:
            print('********************************')
            print("CIFAREnv: Using get_current_batch_with_random_features_2")
            print('********************************')
            self.flag = True
        # Randomly sample a visibility rate for each sample in batch (between 0.1 and 0.9)
        rates = np.random.uniform(0.1, 0.9, size=self.n_envs)
        
        # Create mask with deterministic count but random pattern
        sampled = np.zeros((self.n_envs, self.n_features), dtype=np.uint8)
        
        for i in range(self.n_envs):
            # Deterministic: exact number of patches to make visible
            num_visible = int(self.n_features * rates[i])
            
            # Random pattern: which specific patches are visible
            visible_indices = np.random.permutation(self.n_features)[:num_visible]
            sampled[i, visible_indices] = 1
        
        # Calculate length (number of visible patches per sample)
        length = np.sum(sampled, axis=1).astype(np.int64)
        
        return torch.from_numpy(self.batch_inputs_images), torch.from_numpy(sampled), \
            self.var_labels, torch.from_numpy(length)


    def get_current_batch_with_all_features(self):
        sampled = np.ones((self.n_envs, self.n_features)).astype(int)
        # sampled = (np.random.binomial(1, 0.5, size=(self.n_envs, self.n_features)).astype(int))
        
        # no need for this line
        # batch_obs = np.zeros((self.n_envs, self.n_features, self.n_features + 1),dtype=np.float32).astype(np.float32)

        if self.n_actions == self.n_features + 1:
            length = np.sum(sampled, axis=1).astype(np.int64)
            # x, y = np.where(self.batch_exist)
        else:
             # NOTE this block will not be executed for cifar10 as n_actions = n_features + 1 always
            pass
        
        # NOTE: sampled behaves as self.batch_obs_features except for terminal action
        # NOTE: most likely length wont be required
        return torch.from_numpy(self.batch_inputs_images), torch.from_numpy(sampled), self.var_labels, torch.from_numpy(length)


    def _nonterminal_step(self, actions, nonterminal, length):
        # dont think need length here
        assert self.n_actions == self.n_features + 1
        # as per understanding nonterminal is a boolean array of shape (n_envs,) where True indicates non-terminal actions
        assert np.all(self.batch_acquired[nonterminal, actions[nonterminal]] == 0)
        if np.any(actions[nonterminal] >= self.n_features):
            raise ValueError("Nonterminal mask includes STOP action(s).")
        
        self.batch_acquired[nonterminal, actions[nonterminal]] = 1 # NOTE batch_acquired should not updated from \
                # batch_obs_feature as it is used to calculate available = exist * (1 - acquired) which ensures an already picked action is not picked again
        
        # NO need to updated self.batch_obs_images as it is same as batch_inputs_images always. only batch_obs_feature and batch_acquired needs to be updated
        # self.batch_obs_images[nonterminal] = self.batch_inputs_images[nonterminal]
        self.batch_obs_features[nonterminal, :self.n_features] = self.batch_acquired[nonterminal]

        if self.r_cost.shape:# len(self.r_cost) == self.n_actions - 1:
            self.rewards[nonterminal] = self.r_cost[actions[nonterminal]]
        else:
            #length = self.batch_acquired.sum(axis=-1)
            self.rewards[nonterminal] = self.r_cost

    
    def step(self, actions):
        actions = actions.cpu().numpy()
        actions = actions.reshape(-1)
        length = np.sum(self.batch_acquired, axis=-1).astype(int)
        nonterminal = actions < self.n_actions - 1
        self._nonterminal_step(actions, nonterminal, length) # length is not used _nonterminal_step
        
        if not np.all(nonterminal):
            p_y_logit = self.classify(              # NOTE will need to check if feeding in self.batch_obs_features or self.batch_acquired to classifier
                torch.from_numpy(self.batch_obs_images[~nonterminal]),
                torch.from_numpy(self.batch_acquired[~nonterminal])
            ).detach().cpu().numpy()
            
            reward, prob = self.calc_reward(p_y_logit,
                    self.batch_labels[~nonterminal])
            #### DEBUG #####
            # print(reward.shape)
            # print('reward[0]:', reward[0])
            ################
            self.rewards[~nonterminal] = reward # loglikelihood
            # update terminating episode to new!! (input, exist, label)
            n_terminal = np.sum(~nonterminal)

            if ((self.data.index + n_terminal) % self.data.n_data) < self.data.index:
                self.n_epoch += 1 # TODO check where to use?

            batch = self.data.next_batch(n_terminal) # n_terminal in num (not 128 or n_envs) and output is (images, mask_indices, labels, exist)

            self.batch_inputs_images[~nonterminal] = batch[0] 
            self.batch_inputs_mask_indices[~nonterminal] = batch[1] # must be of shape (n_envs, n_features) or (n_envs, 64)
            self.batch_labels[~nonterminal] = batch[2]
            self.batch_exist[~nonterminal] = batch[3] if batch[3] is not None else np.ones((n_terminal, self.n_features), dtype=np.uint8).astype(np.uint8)

            self.batch_acquired[~nonterminal] = self.batch_inputs_mask_indices[~nonterminal]

            # Not considering the case in the block below as n_actions = n_features + 1 always for cifar10
            # if self.n_actions != self.n_features + 1:
            #     self.batch_acquired_aux[~nonterminal] = 0

            # self.batch_obs[~nonterminal] = 0
            self.batch_obs_images[~nonterminal] = self.batch_inputs_images[~nonterminal]
            self.batch_obs_features[~nonterminal, :self.n_features] = self.batch_inputs_mask_indices[~nonterminal]
            self.batch_obs_features[~nonterminal, self.n_features] = 0  # STOP bit off

        var_dones = torch.from_numpy((~nonterminal).astype(np.uint8))
        return self.var_obs_features, self.var_obs_images, self.var_rewards, var_dones


    def test_step(self, actions, q_a=None):
        actions = actions.cpu().numpy()
        actions = actions.reshape(-1)
        length = np.sum(self.batch_acquired, axis=1).astype(int)

        #  NOTE I don't think q_a is actually used in env class or from env class in main.py
        if self.n_actions == self.n_features + 1:
            self.q_a[length == 0] = 0 #initialize
            self.q_a[np.arange(self.n_envs), length] = q_a #
        else:
            assert False, "Not implemented for n_actions != n_features + 1"

        nonterminal = actions < self.n_actions - 1
        self._nonterminal_step(actions, nonterminal, length)

        if self.n_epoch < 1:
            notignore = np.arange(self.n_envs)
        else:
            notignore = np.delete(np.arange(self.n_envs), self.ignore)
        terminal_ = (~nonterminal)[notignore] # bool indexing

        if not np.any(terminal_): # so if all are nonterminal all Nones are returned
            inputs = acquired = labels = correct = probs = returns = dones = order = None
            if self.n_actions != self.n_features + 1: # will never hit
                acquired_aux = None
            if self.n_classes == 2: # will never hit
                sigmoid_= None
        else: # if some are terminal
            n_terminal = np.sum(terminal_)

            # # DEBUG:
            # print(f"Classifying {n_terminal} terminal episodes")
            # print(f"batch_acquired shape: {self.batch_acquired[notignore][terminal_].shape}")
            # print(f"batch_acquired sample: {self.batch_acquired[notignore][terminal_][0]}")
            # print(f"Sum (features acquired): {self.batch_acquired[notignore][terminal_][0].sum()}")
            p_y_logit = self.classify(   # NOTE will need to check if feeding in self.batch_obs_features or self.batch_acquired to classifier
                torch.from_numpy(self.batch_obs_images[notignore][terminal_]),
                torch.from_numpy(self.batch_acquired[notignore][terminal_])
            ).detach().cpu().numpy()
            

            reward, prob = self.calc_reward(p_y_logit,
                    self.batch_labels[notignore][terminal_])

            # ===================================================================
            # SAVE DEBUG DATA
            # ===================================================================

            # debug_dir = "/gpfs/home/shaana01/CASC-rl/CASC-rl/rl/debug_test_data"
            # os.makedirs(debug_dir, exist_ok=True)

            # batch_id = self.data.index

            # save_dict = {
            #     'images': torch.from_numpy(self.batch_obs_images[notignore][terminal_]),
            #     'acquired': torch.from_numpy(self.batch_acquired[notignore][terminal_]),
            #     'labels': torch.from_numpy(self.batch_labels[notignore][terminal_]),
            #     'predictions': torch.from_numpy(np.argmax(prob, axis=1)),
            #     'probs': torch.from_numpy(prob),
            #     'logits': torch.from_numpy(p_y_logit),
            #     'batch_id': batch_id,
            #     'n_terminal': n_terminal
            # }

            # torch.save(save_dict, os.path.join(debug_dir, f"batch_{batch_id:06d}.pt"))
            # print(f"[DEBUG] Saved batch {batch_id} with {n_terminal} samples")
            # ===================================================================
            
            # NOTE : WILL NOT BE returning order, will need to update main.py
            # tmp = self.batch_obs[notignore][terminal_][:, :, 1:]
            # order = tmp.argmax(axis=2) * tmp.max(axis=2)
            
            # inputs = self.batch_inputs[notignore][terminal_]
            inputs = self.batch_inputs_images[notignore][terminal_]
            acquired = self.batch_acquired[notignore][terminal_]
            labels = self.batch_labels[notignore][terminal_].astype(int)

            if self.n_actions != self.n_features + 1:
                acquired_aux = self.batch_acquired_aux[notignore][terminal_]
            
            if self.n_classes > 2:
                correct = np.equal(np.argmax(prob, 1),
                        self.batch_labels[notignore][terminal_])
                probs = prob[np.arange(len(np.where(terminal_)[0])),labels]
            else:
                sigmoid_ = prob #sigmoid(p_y_logit.reshape(-1))
                correct = np.equal(prob >= 0.5,
                        self.batch_labels[notignore][terminal_])
                probs = np.where(self.batch_labels[notignore][terminal_],
                    prob, 1 - prob)

            returns = self.batch_returns[notignore][terminal_] + reward
            # loglikelihood
            dones = ~nonterminal
            if self.n_epoch >= 1:
                dones[self.ignore] = False
            self.batch_returns[~nonterminal] = 0
            # update terminal episode to new!! (input, exist, label, obs,
            # acquired)
            if self.n_epoch >= 1:
                tmp, = np.where(~nonterminal)
                self.ignore = sorted(list(set(tmp) | set(self.ignore)))
            else:
                remain = self.data.n_data - self.data.index
                self.n_epoch += (1 if remain <= n_terminal else 0)
                n_update = min(remain, n_terminal)
                batch = self.data.next_batch(n_update)
                x, = np.where(~nonterminal)
                where = x[:n_update]
                if self.n_epoch == 1:
                    self.ignore = x[n_update:]
                self.batch_exist[where] = batch[3] if batch[3] is not None \
                    else np.ones((self.n_envs, self.n_features), dtype=np.uint8).astype(np.uint8)
                self.batch_inputs_images[where] = batch[0] 
                self.batch_inputs_mask_indices[where] = batch[1] # must be of shape (n_envs, n_features) or (n_envs, 64)
                self.batch_labels[where] = batch[2]
                self.batch_acquired[where] = self.batch_inputs_mask_indices[where]
                self.batch_obs_features[where, :self.n_features] = self.batch_inputs_mask_indices[where]  # if mask is 64-d
                self.batch_obs_features[where, self.n_features] = 0  # STOP bit off

                # ********************************************************
                # NOTE TESTING
                self.batch_obs_images[where] = batch[0] 
                # ********************************************************

                # if self.n_actions != self.n_features + 1:
                #     self.batch_acquired_aux[where] = 0

        # if self.n_classes > 2:
        #     return self.var_obs,\
        #         (inputs, acquired, labels, correct, probs, returns, dones, order)
        # else:
        #     return self.var_obs,\
        #         (inputs, acquired, labels, correct, probs, returns, dones, order, sigmoid_)

        # NOTE : inputs is from self.batch_inputs_images, order and sigmoid_ removed... need to fix main.py
        return self.var_obs_images, self.var_obs_features, (inputs, acquired, labels, correct, probs, returns, dones)


class ResnetEncoder(nn.Module): # will the model, images and flattened_mask_indices be on gpu already?
    def __init__(self, args): # this is equivalent to batch_obs_images, batch_acquired
        # images and flattened_mask_indices should have same size as batch_obs_images, batch_acquired
        super().__init__()
        self.resnet_model = PreTrainedResNet(args)
        self.resnet_model.eval()  # set to eval mode
        for p in self.resnet_model.parameters():
            p.requires_grad_(False)
        self.patch_size = args.patch_size if hasattr(args, 'patch_size') else 4
        self.mask_ratio = args.mask_ratio if hasattr(args, 'mask_ratio') else 0.2

    def _apply_mask_indices(self, x, flattened_mask_indices): # input must be torch tensors
        # Reshape mask to match image dimensions
        B, C, H, W = x.shape[0], 3, 32, 32 # fixed for cifar10
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size

        if not torch.is_tensor(flattened_mask_indices):
            flattened_mask_indices = torch.from_numpy(flattened_mask_indices)
        mask = flattened_mask_indices.to(device=x.device, dtype=x.dtype)

        mask = mask.reshape(B, 1, num_patches_h, num_patches_w)
        mask = F.interpolate(mask, size=(H, W), mode='nearest')
        # Apply mask
        masked_x = x * mask + 0.0 * (1 - mask)
        return masked_x

    def forward(self, x, flattened_mask_indices):
        with torch.no_grad():  # have added param.requires_grad = False in pretrained_resnet.py so perhaps dont need this, but no harm
            features = self.resnet_model(self._apply_mask_indices(x, flattened_mask_indices))
        return features


class ResnetEncoderFree(nn.Module): # will the model, images and flattened_mask_indices be on gpu already?
    def __init__(self, args): # this is equivalent to batch_obs_images, batch_acquired
        # images and flattened_mask_indices should have same size as batch_obs_images, batch_acquired
        super().__init__()
        self.resnet_model = PreTrainedResNet(args)
        # self.resnet_model.eval()  # set to eval mode
        # for p in self.resnet_model.parameters():
        #     p.requires_grad_(False)
        self.patch_size = args.patch_size if hasattr(args, 'patch_size') else 4
        self.mask_ratio = args.mask_ratio if hasattr(args, 'mask_ratio') else 0.2

    def _apply_mask_indices(self, x, flattened_mask_indices): # input must be torch tensors
        # Reshape mask to match image dimensions
        B, C, H, W = x.shape[0], 3, 32, 32 # fixed for cifar10
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size

        if not torch.is_tensor(flattened_mask_indices):
            flattened_mask_indices = torch.from_numpy(flattened_mask_indices)
        mask = flattened_mask_indices.to(device=x.device, dtype=x.dtype)

        mask = mask.reshape(B, 1, num_patches_h, num_patches_w)
        mask = F.interpolate(mask, size=(H, W), mode='nearest')
        # Apply mask
        masked_x = x * mask + 0.0 * (1 - mask)
        return masked_x

    def forward(self, x, flattened_mask_indices):
        # with torch.no_grad():  # have added param.requires_grad = False in pretrained_resnet.py so perhaps dont need this, but no harm
        #     features = self.resnet_model(self._apply_mask_indices(x, flattened_mask_indices))
        features = self.resnet_model(self._apply_mask_indices(x, flattened_mask_indices))
        return features




def display_demo(config):
    

    # Set up data
    train_dataset = kspaceCIFAR10(config, mode="train")
    val_dataset = kspaceCIFAR10(config, mode="val")
    test_dataset = kspaceCIFAR10(config, mode="test")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True if config.num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True if config.num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True if config.num_workers > 0 else False
    )

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val: {len(val_dataset)}")
    print(f"  Test: {len(test_dataset)}")


    # settig up Mask function
    patch_size = getattr(config, "patch_size", 4)
    mask_ratio = getattr(config, "patch_mask_ratio", 0.5)
    mask_fn = PatchMask(patch_size=patch_size, mask_ratio=mask_ratio, mask_value=0.0,)

    # initialize pretrained resnet
    resnet_model = PreTrainedResNet(config)
    resnet_model.eval()  # set to eval mode

    # iterator over batch(es) from train_loader
    for batch_idx, batch in enumerate(train_loader):
        images = batch['recon_esc']   
        labels = batch['label']['cifar10'].squeeze()

        if batch_idx == 1:  # Just draw one batch for demonstration
            break

        print(f"Batch {batch_idx}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")

        masked_x, flattened_mask_indices, mask = mask_fn(images, return_mask=True)

        with torch.no_grad():  # have added param.requires_grad = False in pretrained_resnet.py so perhaps dont need this, but no harm
            features = resnet_model(masked_x)
        
        print(f"  Masked Images shape: {masked_x.shape}")
        print(f"  Mask shape: {mask.shape}")
        print(f"  Flattened Mask Indices shape: {flattened_mask_indices.shape}")
        print(f"  Features shape: {features.shape}")


        # Visualize masked image and mask for first sample in batch
        #########################################
        # num_patches_h = 8
        # num_patches_w = 8
        # grid_mask_indices = flattened_mask_indices.view(masked_x.shape[0], 1, num_patches_h, num_patches_w)
        # print(grid_mask_indices[0])
        # save_dir = os.path.dirname(os.path.abspath(__file__))
        # img = masked_x[0].detach().cpu()
        # msk = mask[0].detach().cpu()
        # img_path = os.path.join(save_dir, "masked_x_0.png")
        # vutils.save_image(img, img_path, normalize=True)
        # mask_path = os.path.join(save_dir, "mask_0.png")
        # vutils.save_image(msk, mask_path, normalize=True)
        #########################################


def main():
    parser = argparse.ArgumentParser(description='Train baseline CIFAR-10 classifier')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    
    args = parser.parse_args()
    # Load configuration
    config = get_config(args.config)
    display_demo(config)


if __name__ == "__main__":
    main()
