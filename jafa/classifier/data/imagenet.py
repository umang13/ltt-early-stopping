import argparse
import math
import os
from collections import namedtuple
from pathlib import Path

from classifier.utils.transforms import ifftc, fftc
import torchvision
from torchvision import transforms
import torch
from torch.utils.data import Dataset
from torch.utils.data import Subset

# ImageNet normalization values (standard ImageNet normalization)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def imagenet_normalization():
    """ImageNet normalization transform"""
    return transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


# ImageNet class mappings for imbalanced datasets
# These are the actual ImageNet class indices for different categories
IMAGENET_CLASS_MAPPINGS = {
    "dogs": [
        # Dog breeds in ImageNet (class indices 151-268)
        151, 152, 153, 154, 155, 156, 157, 158, 159, 160,  # Various dog breeds
        161, 162, 163, 164, 165, 166, 167, 168, 169, 170,
        171, 172, 173, 174, 175, 176, 177, 178, 179, 180,
        181, 182, 183, 184, 185, 186, 187, 188, 189, 190,
        191, 192, 193, 194, 195, 196, 197, 198, 199, 200,
        201, 202, 203, 204, 205, 206, 207, 208, 209, 210,
        211, 212, 213, 214, 215, 216, 217, 218, 219, 220,
        221, 222, 223, 224, 225, 226, 227, 228, 229, 230,
        231, 232, 233, 234, 235, 236, 237, 238, 239, 240,
        241, 242, 243, 244, 245, 246, 247, 248, 249, 250,
        251, 252, 253, 254, 255, 256, 257, 258, 259, 260,
        261, 262, 263, 264, 265, 266, 267, 268
    ],
    # Specific dog breeds for hierarchical imbalanced datasets
    "german_shepherd": [235],  # German shepherd
    "golden_retriever": [207],  # Golden retriever
    "labrador_retriever": [208],  # Labrador retriever
    "beagle": [162],  # Beagle
    "bulldog": [243],  # Bulldog
    "poodle": [265],  # Poodle
    "rottweiler": [254],  # Rottweiler
    "siberian_husky": [250],  # Siberian husky
    "dachshund": [168],  # Dachshund
    "boxer": [242],  # Boxer
    "cats": [
        # Cat breeds in ImageNet (class indices 281-285)
        281, 282, 283, 284, 285
    ],
    "birds": [
        # Bird species in ImageNet (class indices 80-100)
        80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100
    ],
    "vehicles": [
        # Vehicle classes in ImageNet (selected indices)
        511, 512, 513, 514, 515, 516, 517, 518, 519, 520,  # Cars
        609, 610, 611, 612, 613, 614, 615, 616, 617, 618,  # Trucks
        627, 628, 629, 630, 631, 632, 633, 634, 635, 636,  # Buses
        654, 655, 656, 657, 658, 659, 660, 661, 662, 663,  # Motorcycles
        705, 706, 707, 708, 709, 710, 711, 712, 713, 714,  # Bicycles
        717, 718, 719, 720, 721, 722, 723, 724, 725, 726,  # Airplanes
        730, 731, 732, 733, 734, 735, 736, 737, 738, 739,  # Boats
        740, 741, 742, 743, 744, 745, 746, 747, 748, 749,  # Ships
        750, 751, 752, 753, 754, 755, 756, 757, 758, 759,  # Trains
        760, 761, 762, 763, 764, 765, 766, 767, 768, 769,  # Helicopters
        770, 771, 772, 773, 774, 775, 776, 777, 778, 779,  # Balloons
        780, 781, 782, 783, 784, 785, 786, 787, 788, 789,  # Tanks
        790, 791, 792, 793, 794, 795, 796, 797, 798, 799,  # Tractors
        800, 801, 802, 803, 804, 805, 806, 807, 808, 809,  # Ambulances
        810, 811, 812, 813, 814, 815, 816, 817, 818, 819,  # Fire engines
        820, 821, 822, 823, 824, 825, 826, 827, 828, 829,  # Police cars
        830, 831, 832, 833, 834, 835, 836, 837, 838, 839,  # Limousines
        840, 841, 842, 843, 844, 845, 846, 847, 848, 849,  # Sports cars
        850, 851, 852, 853, 854, 855, 856, 857, 858, 859,  # Convertibles
        860, 861, 862, 863, 864, 865, 866, 867, 868, 869,  # SUVs
        870, 871, 872, 873, 874, 875, 876, 877, 878, 879,  # Pickup trucks
        880, 881, 882, 883, 884, 885, 886, 887, 888, 889,  # Vans
        890, 891, 892, 893, 894, 895, 896, 897, 898, 899,  # Minivans
        900, 901, 902, 903, 904, 905, 906, 907, 908, 909,  # Station wagons
        910, 911, 912, 913, 914, 915, 916, 917, 918, 919,  # Hatchbacks
        920, 921, 922, 923, 924, 925, 926, 927, 928, 929,  # Sedans
        930, 931, 932, 933, 934, 935, 936, 937, 938, 939,  # Coupes
        940, 941, 942, 943, 944, 945, 946, 947, 948, 949,  # Roadsters
        950, 951, 952, 953, 954, 955, 956, 957, 958, 959,  # Cabriolets
        960, 961, 962, 963, 964, 965, 966, 967, 968, 969,  # Phaetons
        970, 971, 972, 973, 974, 975, 976, 977, 978, 979,  # Touring cars
        980, 981, 982, 983, 984, 985, 986, 987, 988, 989,  # Racing cars
        990, 991, 992, 993, 994, 995, 996, 997, 998, 999,  # Formula cars
        1000
    ]
}

def get_imbalanced_class_mapping(imbalanced_class_name):
    """
    Get the class indices for a given imbalanced class name
    
    Args:
        imbalanced_class_name: Name of the imbalanced class (e.g., 'dogs', 'cats', 'birds', 'vehicles')
    
    Returns:
        set: Set of class indices belonging to the imbalanced class
    """
    if imbalanced_class_name not in IMAGENET_CLASS_MAPPINGS:
        raise ValueError(f"Unknown imbalanced class: {imbalanced_class_name}. Available classes: {list(IMAGENET_CLASS_MAPPINGS.keys())}")
    
    return set(IMAGENET_CLASS_MAPPINGS[imbalanced_class_name])


def get_hierarchical_imbalanced_mapping(superclass_name, target_class_name):
    """
    Get the class indices for hierarchical imbalanced datasets
    
    Args:
        superclass_name: Name of the superclass (e.g., 'dogs', 'cats', 'birds', 'vehicles')
        target_class_name: Name of the target class within the superclass (e.g., 'german_shepherd', 'golden_retriever')
    
    Returns:
        tuple: (target_class_indices, other_class_indices) where:
            - target_class_indices: Set of class indices for the target class
            - other_class_indices: Set of class indices for other classes in the superclass
    """
    # Get the superclass indices
    if superclass_name not in IMAGENET_CLASS_MAPPINGS:
        raise ValueError(f"Unknown superclass: {superclass_name}. Available superclasses: {list(IMAGENET_CLASS_MAPPINGS.keys())}")
    
    superclass_indices = set(IMAGENET_CLASS_MAPPINGS[superclass_name])
    
    # Get the target class indices
    if target_class_name not in IMAGENET_CLASS_MAPPINGS:
        raise ValueError(f"Unknown target class: {target_class_name}. Available target classes: {list(IMAGENET_CLASS_MAPPINGS.keys())}")
    
    target_class_indices = set(IMAGENET_CLASS_MAPPINGS[target_class_name])
    
    # Verify that the target class is within the superclass
    if not target_class_indices.issubset(superclass_indices):
        raise ValueError(f"Target class {target_class_name} is not a subset of superclass {superclass_name}")
    
    # Get other classes in the superclass (excluding the target class)
    other_class_indices = superclass_indices - target_class_indices
    
    return target_class_indices, other_class_indices


class ImageNetDataset(Dataset):
    """ImageNet dataset wrapper that provides both image and k-space data"""
    
    def __init__(self, config: argparse.ArgumentParser, mode: str):
        super().__init__()

        self.mode = mode
        self.config = config
        
        # Check if imbalanced dataset is enabled
        self.imbalanced_class = getattr(config, 'imbalanced_class', None)
        self.is_imbalanced = self.imbalanced_class is not None
        
        if self.is_imbalanced:
            print(f"Initializing imbalanced ImageNet dataset for class: {self.imbalanced_class}")
            self.imbalanced_class_indices = get_imbalanced_class_mapping(self.imbalanced_class)
            print(f"Imbalanced class contains {len(self.imbalanced_class_indices)} ImageNet classes")
        
        # ImageNet standard transforms (without normalization - handled in forward method)
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        
        self.test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

        self.prepare_data()

    def prepare_data(self):
        if self.mode == "train":
            self.data_split = "train"
            # For ImageNet, we typically use a subset for training due to size
            # You can modify this based on your needs
            self.dataset = torchvision.datasets.ImageFolder(
                root=os.path.join(self.config.datadir, "train"),
                transform=self.transform,
            )
            
        elif self.mode == "val":
            self.data_split = "val"
            self.dataset = torchvision.datasets.ImageFolder(
                root=os.path.join(self.config.datadir, "val"),
                transform=self.test_transform,
            )

        elif self.mode == "test":
            self.data_split = "test"
            self.dataset = torchvision.datasets.ImageFolder(
                root=os.path.join(self.config.datadir, "val"),  # Use val as test
                transform=self.test_transform,
            )
        else:
            raise ValueError(f"Invalid mode {self.mode}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]

        # Handle imbalanced dataset: convert to binary classification
        if self.is_imbalanced:
            # Binary label: 1 if belongs to imbalanced class, 0 otherwise
            binary_label = 1 if label in self.imbalanced_class_indices else 0
            label = binary_label

        # Compute k-space for consistency
        kspace = torch.fft.fft2(image)  # Perform FFT
        kspace_arr = torch.fft.fftshift(kspace, dim=(-2, -1))
        
        parameters = {
            "sc_kspace": kspace_arr,  # Always provide k-space data
            "recon_esc": image,       # Always provide image data
            "sampled_indices": (0, -1),
            "data_split": self.data_split,
            "dataset": self.config.dataset,
            "max_value": kspace_arr.abs().max(),
            "label": dict(imagenet=torch.tensor([label]).long()),
        }

        return parameters


class kspaceImageNet(Dataset):
    """K-space ImageNet dataset with variable density sampling"""
    
    def __init__(self, config: argparse.ArgumentParser, mode: str):
        super().__init__()

        self.mode = mode
        self.config = config
        
        # Check if imbalanced dataset is enabled
        self.imbalanced_class = getattr(config, 'imbalanced_class', None)
        self.is_imbalanced = self.imbalanced_class is not None
        
        if self.is_imbalanced:
            print(f"Initializing imbalanced k-space ImageNet dataset for class: {self.imbalanced_class}")
            self.imbalanced_class_indices = get_imbalanced_class_mapping(self.imbalanced_class)
            print(f"Imbalanced class contains {len(self.imbalanced_class_indices)} ImageNet classes")
        
        # ImageNet standard transforms (without normalization - handled in forward method)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        
        self.test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            # transforms.RandomCrop(224, padding=32),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        self.prepare_data()

    def prepare_data(self):
        if self.mode == "train":
            self.data_split = "train"
            self.dataset = torchvision.datasets.ImageFolder(
                root=os.path.join(self.config.datadir, "train"),
                transform=self.transform,
            )
            
        elif self.mode == "val":
            self.data_split = "val"
            self.dataset = torchvision.datasets.ImageFolder(
                root=os.path.join(self.config.datadir, "val"),
                transform=self.test_transform,
            )

        elif self.mode == "test":
            self.data_split = "test"
            self.dataset = torchvision.datasets.ImageFolder(
                root=os.path.join(self.config.datadir, "val"),  # Use val as test
                transform=self.test_transform,
            )
        else:
            raise ValueError(f"Invalid mode {self.mode}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]

        # Handle imbalanced dataset: convert to binary classification
        if self.is_imbalanced:
            # Binary label: 1 if belongs to imbalanced class, 0 otherwise
            binary_label = 1 if label in self.imbalanced_class_indices else 0
            label = binary_label
            
            # Add debugging information (only print occasionally to avoid spam)
            if hasattr(self, '_debug_counter'):
                self._debug_counter += 1
            else:
                self._debug_counter = 0
                
            if self._debug_counter < 10:  # Only print first 10 samples
                print(f"kspaceImageNet Sample {self._debug_counter}: Original label {label} -> Binary label {binary_label} (class: {self.imbalanced_class})")

        # Compute k-space for consistency
        kspace = torch.fft.fft2(image)  # Perform FFT
        kspace_arr = torch.fft.fftshift(kspace, dim=(-2, -1))
        # print("kspace_arr shape: ", kspace_arr.shape)
        # print("image shape: ", image.shape)
        
        parameters = {
            "sc_kspace": kspace_arr,  # Always provide k-space data
            "recon_esc": image,       # Always provide image data
            "sampled_indices": (0, -1),
            "data_split": self.data_split,
            "dataset": self.config.dataset,
            "max_value": kspace_arr.abs().max(),
            "label": dict(imagenet=torch.tensor([label]).long()),
        }

        return parameters


class HierarchicalImbalancedImageNet(Dataset):
    """
    Hierarchical imbalanced ImageNet dataset
    
    First selects only images from a majority superclass (e.g., dogs), then creates
    an imbalanced dataset within that superclass (e.g., 'german shepherd' vs 'all other dogs')
    """
    
    def __init__(self, config: argparse.ArgumentParser, mode: str):
        super().__init__()

        self.mode = mode
        self.config = config
        
        # Check if hierarchical imbalanced dataset is enabled
        self.superclass = getattr(config, 'hierarchical_superclass', None)
        self.target_class = getattr(config, 'hierarchical_target_class', None)
        self.is_hierarchical_imbalanced = self.superclass is not None and self.target_class is not None
        
        if self.is_hierarchical_imbalanced:
            print(f"Initializing hierarchical imbalanced ImageNet dataset:")
            print(f"  Superclass: {self.superclass}")
            print(f"  Target class: {self.target_class}")
            
            # Get the class mappings
            self.target_class_indices, self.other_class_indices = get_hierarchical_imbalanced_mapping(
                self.superclass, self.target_class
            )
            
            print(f"  Target class contains {len(self.target_class_indices)} ImageNet classes: {self.target_class_indices}")
            print(f"  Other classes contain {len(self.other_class_indices)} ImageNet classes")
        
        # ImageNet standard transforms (without normalization - handled in forward method)
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        
        self.test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

        self.prepare_data()
        self.filter_dataset()

    def prepare_data(self):
        if self.mode == "train":
            self.data_split = "train"
            self.dataset = torchvision.datasets.ImageFolder(
                root=os.path.join(self.config.datadir, "train"),
                transform=self.transform,
            )
            
        elif self.mode == "val":
            self.data_split = "val"
            self.dataset = torchvision.datasets.ImageFolder(
                root=os.path.join(self.config.datadir, "val"),
                transform=self.test_transform,
            )

        elif self.mode == "test":
            self.data_split = "test"
            self.dataset = torchvision.datasets.ImageFolder(
                root=os.path.join(self.config.datadir, "val"),  # Use val as test
                transform=self.test_transform,
            )
        else:
            raise ValueError(f"Invalid mode {self.mode}")

    def filter_dataset(self):
        """Filter dataset to only include samples from the superclass"""
        if self.is_hierarchical_imbalanced:
            # Get indices of samples that belong to the superclass
            superclass_indices = self.target_class_indices | self.other_class_indices
            valid_indices = []
            
            for idx, (_, label) in enumerate(self.dataset.samples):
                if label in superclass_indices:
                    valid_indices.append(idx)
            
            # Create a subset with only valid samples
            from torch.utils.data import Subset
            original_len = len(self.dataset)
            self.dataset = Subset(self.dataset, valid_indices)
            
            print(f"Filtered dataset: {len(valid_indices)} samples from superclass '{self.superclass}' out of {original_len} total samples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]

        # Handle hierarchical imbalanced dataset: convert to binary classification within superclass
        if self.is_hierarchical_imbalanced:
            # Check if the sample belongs to the superclass
            superclass_indices = self.target_class_indices | self.other_class_indices
            if label in superclass_indices:
                # Binary label: 1 if belongs to target class, 0 if belongs to other classes in superclass
                binary_label = 1 if label in self.target_class_indices else 0
                label = binary_label
            else:
                # For samples that don't belong to the superclass, we should not reach here
                # because the dataset should be pre-filtered. If we do reach here, it's an error.
                raise ValueError(f"Sample with label {label} not in superclass {self.superclass}. Dataset should be pre-filtered.")

        # Compute k-space for consistency
        kspace = torch.fft.fft2(image)  # Perform FFT
        kspace_arr = torch.fft.fftshift(kspace, dim=(-2, -1))
        
        parameters = {
            "sc_kspace": kspace_arr,  # Always provide k-space data
            "recon_esc": image,       # Always provide image data
            "sampled_indices": (0, -1),
            "data_split": self.data_split,
            "dataset": self.config.dataset,
            "max_value": kspace_arr.abs().max(),
            "label": dict(imagenet=torch.tensor([label]).long()),
        }

        return parameters


class HierarchicalImbalancedKspaceImageNet(Dataset):
    """
    Hierarchical imbalanced k-space ImageNet dataset with variable density sampling
    
    First selects only images from a majority superclass (e.g., dogs), then creates
    an imbalanced dataset within that superclass (e.g., 'german shepherd' vs 'all other dogs')
    """
    
    def __init__(self, config: argparse.ArgumentParser, mode: str):
        super().__init__()

        self.mode = mode
        self.config = config
        
        # Check if hierarchical imbalanced dataset is enabled
        self.superclass = getattr(config, 'hierarchical_superclass', None)
        self.target_class = getattr(config, 'hierarchical_target_class', None)
        self.is_hierarchical_imbalanced = self.superclass is not None and self.target_class is not None
        
        if self.is_hierarchical_imbalanced:
            print(f"Initializing hierarchical imbalanced k-space ImageNet dataset:")
            print(f"  Superclass: {self.superclass}")
            print(f"  Target class: {self.target_class}")
            
            # Get the class mappings
            self.target_class_indices, self.other_class_indices = get_hierarchical_imbalanced_mapping(
                self.superclass, self.target_class
            )
            
            print(f"  Target class contains {len(self.target_class_indices)} ImageNet classes: {self.target_class_indices}")
            print(f"  Other classes contain {len(self.other_class_indices)} ImageNet classes")
        
        # ImageNet standard transforms (without normalization - handled in forward method)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        
        self.test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

        self.prepare_data()
        self.filter_dataset()

    def prepare_data(self):
        if self.mode == "train":
            self.data_split = "train"
            self.dataset = torchvision.datasets.ImageFolder(
                root=os.path.join(self.config.datadir, "train"),
                transform=self.transform,
            )
            
        elif self.mode == "val":
            self.data_split = "val"
            self.dataset = torchvision.datasets.ImageFolder(
                root=os.path.join(self.config.datadir, "val"),
                transform=self.test_transform,
            )

        elif self.mode == "test":
            self.data_split = "test"
            self.dataset = torchvision.datasets.ImageFolder(
                root=os.path.join(self.config.datadir, "val"),  # Use val as test
                transform=self.test_transform,
            )
        else:
            raise ValueError(f"Invalid mode {self.mode}")

    def filter_dataset(self):
        """Filter dataset to only include samples from the superclass"""
        if self.is_hierarchical_imbalanced:
            # Get indices of samples that belong to the superclass
            superclass_indices = self.target_class_indices | self.other_class_indices
            valid_indices = []
            
            for idx, (_, label) in enumerate(self.dataset.samples):
                if label in superclass_indices:
                    valid_indices.append(idx)
            
            # Create a subset with only valid samples
            from torch.utils.data import Subset
            original_len = len(self.dataset)
            self.dataset = Subset(self.dataset, valid_indices)
            
            print(f"Filtered dataset: {len(valid_indices)} samples from superclass '{self.superclass}' out of {original_len} total samples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]

        # Handle hierarchical imbalanced dataset: convert to binary classification within superclass
        if self.is_hierarchical_imbalanced:
            # Check if the sample belongs to the superclass
            superclass_indices = self.target_class_indices | self.other_class_indices
            if label in superclass_indices:
                # Binary label: 1 if belongs to target class, 0 if belongs to other classes in superclass
                binary_label = 1 if label in self.target_class_indices else 0
                label = binary_label
                
                # Add debugging information (only print occasionally to avoid spam)
                if hasattr(self, '_debug_counter'):
                    self._debug_counter += 1
                else:
                    self._debug_counter = 0
                    
                if self._debug_counter < 10:  # Only print first 10 samples
                    print(f"HierarchicalImbalancedKspaceImageNet Sample {self._debug_counter}: Original label {label} -> Binary label {binary_label} (superclass: {self.superclass}, target: {self.target_class})")
            else:
                # For samples that don't belong to the superclass, we should not reach here
                # because the dataset should be pre-filtered. If we do reach here, it's an error.
                raise ValueError(f"Sample with label {label} not in superclass {self.superclass}. Dataset should be pre-filtered.")

        # Compute k-space for consistency
        kspace = torch.fft.fft2(image)  # Perform FFT
        kspace_arr = torch.fft.fftshift(kspace, dim=(-2, -1))
        
        parameters = {
            "sc_kspace": kspace_arr,  # Always provide k-space data
            "recon_esc": image,       # Always provide image data
            "sampled_indices": (0, -1),
            "data_split": self.data_split,
            "dataset": self.config.dataset,
            "max_value": kspace_arr.abs().max(),
            "label": dict(imagenet=torch.tensor([label]).long()),
        }

        return parameters