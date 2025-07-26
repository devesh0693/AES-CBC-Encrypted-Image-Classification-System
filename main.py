import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from PIL import Image
import customtkinter as ctk
from tkinter import filedialog, messagebox, simpledialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time
import hashlib
import hmac
import secrets
import getpass
from pathlib import Path

from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Hash import SHA256
import base64

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class SecureKeyManager:
    def __init__(self):
        self.key_derivation_iterations = 100000 
        self.salt_size = 32 
        self.key_size = 32 

    def derive_key_from_password(self, password: str, salt: bytes = None) -> tuple:
        if salt == None:
            salt = get_random_bytes(self.salt_size)

        key = PBKDF2(
            password.encode('utf-8'),
            salt,
            dkLen=self.key_size, 
            count=self.key_derivation_iterations, 
            hmac_hash_module=SHA256 
        )

        return key, salt

    def generate_secure_key(self) -> bytes:
        return secrets.token_bytes(self.key_size)

    def compute_key_hash(self, key: bytes) -> str:
        return hashlib.sha256(key).hexdigest()[:16]

    def validate_password_strength(self, password: str) -> tuple:
        issues = []
        if len(password) < 12:
            issues.append("Password must be at least 12 characters long")
        if not any(c.isupper() for c in password):
            issues.append("Password must contain uppercase letters")
        if not any(c.islower() for c in password):
            issues.append("Password must contain lowercase letters")
        if not any(c.isdigit() for c in password):
            issues.append("Password must contain numbers")
        if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            issues.append("Password must contain special characters")

        return len(issues) == 0, issues

class AdvancedAESProcessor:
    def __init__(self, key_size=32):
        self.key_size = key_size 
        self.block_size = AES.block_size 
        self.current_key = None

    def encrypt_image_structure_preserving(self, image_array: np.ndarray) -> tuple:
        if self.current_key is None:
            raise ValueError("Encryption key is not set in AdvancedAESProcessor.")
        
        original_shape = image_array.shape
        height, width, channels = original_shape
        
        iv = get_random_bytes(AES.block_size)
        cipher = AES.new(self.current_key, AES.MODE_CBC, iv)
        
        encrypted_channels = []
        
        for c in range(channels):
            channel = image_array[:, :, c].astype(np.uint8)
            channel_bytes = channel.tobytes()
            
            padded_data = pad(channel_bytes, AES.block_size)
            encrypted_channel_bytes = cipher.encrypt(padded_data)
            
            cipher = AES.new(self.current_key, AES.MODE_CBC, iv)
            
            encrypted_channel = np.frombuffer(encrypted_channel_bytes, dtype=np.uint8)
            
            if len(encrypted_channel) > height * width:
                encrypted_channel = encrypted_channel[:height * width]
            elif len(encrypted_channel) < height * width:
                padding_needed = height * width - len(encrypted_channel)
                encrypted_channel = np.concatenate([
                    encrypted_channel, 
                    np.zeros(padding_needed, dtype=np.uint8)
                ])
            
            encrypted_channel = encrypted_channel.reshape(height, width)
            encrypted_channels.append(encrypted_channel)
        
        encrypted_image = np.stack(encrypted_channels, axis=2)
        
        encrypted_bytes = encrypted_image.tobytes()
        
        return iv + encrypted_bytes, iv

    def decrypt_image_structure_preserving(self, encrypted_data: bytes, original_shape: tuple) -> np.ndarray:
        if self.current_key is None:
            raise ValueError("Decryption key is not set in AdvancedAESProcessor.")
        
        height, width, channels = original_shape
        
        iv = encrypted_data[:AES.block_size]
        encrypted_bytes = encrypted_data[AES.block_size:]
        
        encrypted_image = np.frombuffer(encrypted_bytes, dtype=np.uint8).reshape(original_shape)
        
        decrypted_channels = []
        
        for c in range(channels):
            encrypted_channel = encrypted_image[:, :, c]
            encrypted_channel_bytes = encrypted_channel.tobytes()
            
            cipher = AES.new(self.current_key, AES.MODE_CBC, iv)
            
            original_channel_size = height * width
            padding_needed = (AES.block_size - (original_channel_size % AES.block_size)) % AES.block_size
            padded_size = original_channel_size + padding_needed
            
            if len(encrypted_channel_bytes) < padded_size:
                padding_bytes = padded_size - len(encrypted_channel_bytes)
                encrypted_channel_bytes += b'\x00' * padding_bytes
            elif len(encrypted_channel_bytes) > padded_size:
                encrypted_channel_bytes = encrypted_channel_bytes[:padded_size]
            
            try:
                decrypted_padded = cipher.decrypt(encrypted_channel_bytes)
                decrypted_channel_bytes = unpad(decrypted_padded, AES.block_size)
            except ValueError:
                decrypted_channel_bytes = decrypted_padded[:original_channel_size]
            
            if len(decrypted_channel_bytes) > original_channel_size:
                decrypted_channel_bytes = decrypted_channel_bytes[:original_channel_size]
            elif len(decrypted_channel_bytes) < original_channel_size:
                padding_needed = original_channel_size - len(decrypted_channel_bytes)
                decrypted_channel_bytes += b'\x00' * padding_needed
            
            decrypted_channel = np.frombuffer(decrypted_channel_bytes, dtype=np.uint8).reshape(height, width)
            decrypted_channels.append(decrypted_channel)
        
        decrypted_image = np.stack(decrypted_channels, axis=2)
        
        return decrypted_image

    def encrypt_image_cbc(self, image_array: np.ndarray) -> tuple:
        if self.current_key is None:
            raise ValueError("Encryption key is not set in AdvancedAESProcessor.")
        
        original_shape = image_array.shape
        
        iv = get_random_bytes(AES.block_size)
        cipher = AES.new(self.current_key, AES.MODE_CBC, iv)
        
        image_bytes = image_array.astype(np.uint8).tobytes()
        padded_data = pad(image_bytes, AES.block_size)
        encrypted_data = cipher.encrypt(padded_data)
        
        return iv + encrypted_data, iv

    def decrypt_image_cbc(self, encrypted_data: bytes, original_shape: tuple) -> np.ndarray:
        if self.current_key is None:
            raise ValueError("Decryption key is not set in AdvancedAESProcessor.")
        
        iv = encrypted_data[:AES.block_size]
        ciphertext = encrypted_data[AES.block_size:]
        
        cipher = AES.new(self.current_key, AES.MODE_CBC, iv)
        decrypted_padded = cipher.decrypt(ciphertext)
        
        try:
            decrypted_data = unpad(decrypted_padded, AES.block_size)
        except ValueError:
            expected_size = np.prod(original_shape)
            decrypted_data = decrypted_padded[:expected_size]
        
        expected_size = np.prod(original_shape)
        if len(decrypted_data) > expected_size:
            decrypted_data = decrypted_data[:expected_size]
        elif len(decrypted_data) < expected_size:
            padding_needed = expected_size - len(decrypted_data)
            decrypted_data += b'\x00' * padding_needed
        
        decrypted_image = np.frombuffer(decrypted_data, dtype=np.uint8).reshape(original_shape)
        
        return decrypted_image

    def set_key(self, key: bytes):
        self.current_key = key

class HybridDataset(Dataset):
    def __init__(self, original_data, encrypted_data, labels, encryption_metadata, mode='original'):
        self.original_data = original_data
        self.encrypted_data = encrypted_data
        self.labels = labels 
        self.encryption_metadata = encryption_metadata 
        self.mode = mode
        self.data_integrity_hash = self._compute_dataset_hash()
        
        self.transform_original = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
        ])
        
        self.transform_encrypted = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _compute_dataset_hash(self) -> str:
        hasher = hashlib.sha256()
        for data in self.encrypted_data:
            if isinstance(data, bytes):
                hasher.update(data)
            else:
                hasher.update(str(data).encode())
        return hasher.hexdigest()

    def verify_integrity(self) -> bool:
        current_hash = self._compute_dataset_hash()
        return current_hash == self.data_integrity_hash

    def set_mode(self, mode):
        if mode in ['original', 'encrypted']:
            self.mode = mode
        else:
            raise ValueError("Mode must be 'original' or 'encrypted'")

    def __len__(self):
        return len(self.original_data)

    def __getitem__(self, idx):
        label = self.labels[idx]

        if self.mode == 'original':
            image = self.original_data[idx]
            if isinstance(image, np.ndarray):
                image_tensor = self.transform_original(image)
            else:
                image_tensor = image
        else:
            if isinstance(self.encrypted_data[idx], np.ndarray):
                encrypted_img = self.encrypted_data[idx]
                image_tensor = self.transform_encrypted(encrypted_img)
            else:
                encrypted_img = self.encrypted_data[idx]
                if len(encrypted_img) > 16:
                    encrypted_array = np.frombuffer(encrypted_img[16:], dtype=np.uint8)
                    if len(encrypted_array) >= 32*32*3:
                        encrypted_array = encrypted_array[:32*32*3].reshape(32, 32, 3)
                        image_tensor = self.transform_encrypted(encrypted_array)
                    else:
                        image_tensor = torch.zeros(3, 32, 32)
                else:
                    image_tensor = torch.zeros(3, 32, 32)

        return image_tensor, label

class OptimizedCNN(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.3):
        super(OptimizedCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate * 0.3),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate * 0.4),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.3),
            nn.Linear(256, num_classes)
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class SecureImageClassifierApp:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("Encrypted Image Classification (AES-CBC) v3.0 - By Devesh Rawat")
        self.root.geometry("1400x900")
        self.root.resizable(True, True)

        self.key_manager = SecureKeyManager()
        self.aes_processor = AdvancedAESProcessor()

        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encryption_key = None
        self.key_hash = None
        self.encryption_mode = "AES-CBC"
        self.session_authenticated = False

        self.is_training = False
        self.training_thread = None
        self.training_metrics = {"losses": [], "accuracies": [], "val_accuracies": []}

        self.train_dataset = None
        self.test_dataset = None
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                           'dog', 'frog', 'horse', 'ship', 'truck']

        self.setup_ui()
        self._update_info_labels()
        self.log_message("AES-CBC Encrypted Image Classification System Initialized")
        self.log_message(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            self.log_message("CUDA is available. Optimized training enabled.")
        else:
            self.log_message("Using CPU. Consider using GPU for faster training.")
        self.log_message("Please authenticate to begin secure operations")

    def setup_ui(self):
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        main_frame = ctk.CTkFrame(self.root)
        main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(2, weight=1)

        self.setup_title_section(main_frame)
        self.setup_control_panel(main_frame)
        self.setup_main_content(main_frame)
        self.setup_status_bar(main_frame)

    def setup_title_section(self, parent):
        title_frame = ctk.CTkFrame(parent)
        title_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        title_frame.grid_columnconfigure(1, weight=1)

        title_label = ctk.CTkLabel(title_frame, text="AES-CBC Encrypted Image Classification",
                                  font=ctk.CTkFont(size=24, weight="bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=10)

        self.security_status = ctk.CTkLabel(title_frame, text="Not Authenticated",
                                          font=ctk.CTkFont(size=14))
        self.security_status.grid(row=1, column=0, padx=10, pady=5, sticky="w")

        self.encryption_mode_label = ctk.CTkLabel(title_frame, text=f"Architecture: Optimized CNN | Encryption: AES-CBC",
                                                 font=ctk.CTkFont(size=12))
        self.encryption_mode_label.grid(row=1, column=1, padx=10, pady=5, sticky="e")

    def setup_control_panel(self, parent):
        control_frame = ctk.CTkFrame(parent)
        control_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)

        security_frame = ctk.CTkFrame(control_frame)
        security_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(security_frame, text="Security Controls",
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)
        security_buttons = ctk.CTkFrame(security_frame)
        security_buttons.pack(fill="x", padx=10, pady=5)
        
        self.auth_btn = ctk.CTkButton(security_buttons, text="Authenticate", command=self.authenticate_user)
        self.auth_btn.pack(side="left", padx=5)
        self.key_gen_btn = ctk.CTkButton(security_buttons, text="Generate Key", command=self.generate_encryption_key, state="disabled")
        self.key_gen_btn.pack(side="left", padx=5)
        self.key_from_pwd_btn = ctk.CTkButton(security_buttons, text="Key from Password", command=self.create_key_from_password, state="disabled")
        self.key_from_pwd_btn.pack(side="left", padx=5)

        ml_frame = ctk.CTkFrame(control_frame)
        ml_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(ml_frame, text="Machine Learning Controls",
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)
        ml_buttons = ctk.CTkFrame(ml_frame)
        ml_buttons.pack(fill="x", padx=10, pady=5)
        
        self.load_data_btn = ctk.CTkButton(ml_buttons, text="Load Dataset", command=self.load_dataset, state="disabled")
        self.load_data_btn.pack(side="left", padx=5)
        self.train_btn = ctk.CTkButton(ml_buttons, text="Train Model", command=self.start_training, state="disabled")
        self.train_btn.pack(side="left", padx=5)
        self.classify_btn = ctk.CTkButton(ml_buttons, text="Classify Image", command=self.classify_image, state="disabled")
        self.classify_btn.pack(side="left", padx=5)

        self.mode_var = ctk.StringVar(value="original")
        mode_frame = ctk.CTkFrame(ml_buttons)
        mode_frame.pack(side="right", padx=5)
        ctk.CTkLabel(mode_frame, text="Training Mode:").pack(side="left", padx=2)
        mode_menu = ctk.CTkOptionMenu(mode_frame, variable=self.mode_var, 
                                     values=["original", "encrypted"], 
                                     command=self.on_mode_change)
        mode_menu.pack(side="left", padx=2)

        model_buttons = ctk.CTkFrame(ml_buttons)
        model_buttons.pack(side="right", padx=5)
        self.save_model_btn = ctk.CTkButton(model_buttons, text="Save Model", command=self.save_model, state="disabled")
        self.save_model_btn.pack(side="left", padx=2)
        self.load_model_btn = ctk.CTkButton(model_buttons, text="Load Model", command=self.load_model, state="normal")
        self.load_model_btn.pack(side="left", padx=2)

    def setup_main_content(self, parent):
        content_frame = ctk.CTkFrame(parent)
        content_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=5)
        content_frame.grid_columnconfigure(0, weight=1)
        content_frame.grid_columnconfigure(1, weight=1)
        content_frame.grid_rowconfigure(0, weight=1)

        left_panel = ctk.CTkFrame(content_frame)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        left_panel.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(left_panel, text="System Log", font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, pady=5)
        self.log_textbox = ctk.CTkTextbox(left_panel, height=300, font=ctk.CTkFont(size=11))
        self.log_textbox.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        info_panel = ctk.CTkFrame(left_panel)
        info_panel.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        self.system_info = ctk.CTkLabel(info_panel, text="System Ready", font=ctk.CTkFont(size=10))
        self.system_info.pack(pady=5)

        right_panel = ctk.CTkScrollableFrame(content_frame, label_text="Training Analytics", 
                                           label_font=ctk.CTkFont(size=16, weight="bold"))
        right_panel.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        right_panel.grid_columnconfigure(0, weight=1)

        progress_frame = ctk.CTkFrame(right_panel)
        progress_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        self.progress_bar = ctk.CTkProgressBar(progress_frame)
        self.progress_bar.pack(fill="x", padx=10, pady=5)
        self.progress_bar.set(0)
        self.progress_label = ctk.CTkLabel(progress_frame, text="Ready")
        self.progress_label.pack(pady=5)
        self.metrics_text = ctk.CTkTextbox(progress_frame, height=120, font=ctk.CTkFont(size=10))
        self.metrics_text.pack(fill="x", expand=False, padx=5, pady=5)

        self.plot_frame = ctk.CTkFrame(right_panel)
        self.plot_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)

        info_display_frame = ctk.CTkFrame(right_panel)
        info_display_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=10)
        info_display_frame.grid_columnconfigure((0, 1), weight=1)

        sec_info_frame = ctk.CTkFrame(info_display_frame)
        sec_info_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        ctk.CTkLabel(sec_info_frame, text="Security Info", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=5)
        self.security_info_label = ctk.CTkLabel(sec_info_frame, text="", justify="left", font=ctk.CTkFont(size=11))
        self.security_info_label.pack(pady=5, padx=10, fill="x")

        model_info_frame = ctk.CTkFrame(info_display_frame)
        model_info_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        ctk.CTkLabel(model_info_frame, text="Model Info", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=5)
        self.model_info_label = ctk.CTkLabel(model_info_frame, text="", justify="left", font=ctk.CTkFont(size=11))
        self.model_info_label.pack(pady=5, padx=10, fill="x")

    def setup_status_bar(self, parent):
        status_frame = ctk.CTkFrame(parent)
        status_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=5)
        self.status_label = ctk.CTkLabel(status_frame, text="Ready - Please authenticate to begin",
                                        font=ctk.CTkFont(size=12))
        self.status_label.pack(side="left", padx=10, pady=5)
        self.connection_indicator = ctk.CTkLabel(status_frame, text="Disconnected",
                                               font=ctk.CTkFont(size=10))
        self.connection_indicator.pack(side="right", padx=10, pady=5)

    def _update_info_labels(self):
        sec_text = (f"Encryption: {self.encryption_mode}\n"
                   f"Key Hash: {self.key_hash or 'None'}\n"
                   f"Training Mode: {self.mode_var.get()}")
        self.security_info_label.configure(text=sec_text)

        if hasattr(self, 'model') and self.model:
            total_params = sum(p.numel() for p in self.model.parameters())
            model_info = (f'Architecture: Optimized CNN\n'
                         f'Total Parameters: {total_params:,}\n'
                         f'Device: {self.device}')
        else:
            model_info = "No model loaded/trained."
        self.model_info_label.configure(text=model_info)

    def log_message(self, message):
        timestamp = time.strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
        self.log_textbox.insert("end", formatted_message)
        self.log_textbox.see("end")
        self.root.update_idletasks()

    def authenticate_user(self):
        dialog = ctk.CTkInputDialog(text="Enter master password:", title="Authentication")
        password = dialog.get_input()
        if password:
            is_strong, issues = self.key_manager.validate_password_strength(password)
            if is_strong:
                self.session_authenticated = True
                self.security_status.configure(text="Authenticated")
                self.connection_indicator.configure(text="Connected")
                self.key_gen_btn.configure(state="normal")
                self.key_from_pwd_btn.configure(state="normal")
                self.load_model_btn.configure(state="normal")
                self.load_data_btn.configure(state="normal")
                self.log_message("User authenticated successfully")
                self.status_label.configure(text="Authenticated - Ready for secure operations")
            else:
                messagebox.showerror("Weak Password", "Password requirements:\n" + "\n".join(issues))
                self.log_message("Authentication failed - weak password")
        else:
            self.log_message("Authentication cancelled")

    def generate_encryption_key(self):
        if not self.session_authenticated:
            messagebox.showerror("Error", "Please authenticate first!")
            return
        try:
            self.encryption_key = self.key_manager.generate_secure_key()
            self.key_hash = self.key_manager.compute_key_hash(self.encryption_key)
            self.aes_processor.set_key(self.encryption_key)
            self.log_message(f"Secure AES-CBC encryption key generated (Hash: {self.key_hash})")
            self.status_label.configure(text="AES-CBC encryption key ready")
            self._update_info_labels()
            self.save_model_btn.configure(state="normal")
            if self.model:
                self.classify_btn.configure(state="normal")
        except Exception as e:
            self.log_message(f"Key generation failed: {str(e)}")
            messagebox.showerror("Error", f"Failed to generate key: {str(e)}")

    def create_key_from_password(self):
        if not self.session_authenticated:
            messagebox.showerror("Error", "Please authenticate first!")
            return
        dialog = ctk.CTkInputDialog(text="Enter password for key derivation:", title="Key Derivation")
        password = dialog.get_input()
        if password:
            try:
                is_strong, issues = self.key_manager.validate_password_strength(password)
                if not is_strong:
                    messagebox.showwarning("Weak Password", "Consider using a stronger password:\n" + "\n".join(issues))
                self.encryption_key, _ = self.key_manager.derive_key_from_password(password)
                self.key_hash = self.key_manager.compute_key_hash(self.encryption_key)
                self.aes_processor.set_key(self.encryption_key)
                self.log_message(f"AES-CBC key derived from password (Hash: {self.key_hash})")
                self.status_label.configure(text="AES-CBC encryption key ready from password")
                self._update_info_labels()
                self.save_model_btn.configure(state="normal")
                if self.model:
                    self.classify_btn.configure(state="normal")
            except Exception as e:
                self.log_message(f"Key derivation failed: {str(e)}")
                messagebox.showerror("Error", f"Failed to derive key: {str(e)}")

    def on_mode_change(self, value):
        if hasattr(self, 'train_dataset') and self.train_dataset:
            self.train_dataset.set_mode(value)
            self.test_dataset.set_mode(value)
            self.log_message(f"Training mode changed to: {value}")
            self._update_info_labels()

    def load_dataset(self):
        if not self.session_authenticated:
            messagebox.showerror("Error", "Please authenticate first!")
            return
            
        self.log_message("Loading CIFAR-10 dataset...")
        self.status_label.configure(text="Loading dataset...")
        self.progress_bar.set(0)
        
        try:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: (x * 255).byte().numpy().transpose(1, 2, 0))
            ])
            
            train_dataset_raw = CIFAR10(root='./data', train=True, download=True, transform=transform)
            test_dataset_raw = CIFAR10(root='./data', train=False, download=True, transform=transform)
            
            train_size = min(15000, len(train_dataset_raw))
            test_size = min(3000, len(test_dataset_raw))
            
            train_data_original, train_data_encrypted, train_labels = [], [], []
            test_data_original, test_data_encrypted, test_labels = [], [], []
            
            encryption_metadata = {"mode": self.encryption_mode, "key_hash": self.key_hash}

            self.log_message(f"Processing {train_size} training images with AES-CBC encryption...")
            for i in range(train_size):
                image_array, label = train_dataset_raw[i]
                train_data_original.append(image_array)
                train_labels.append(label)
                
                if self.encryption_key:
                    encrypted_data, _ = self.aes_processor.encrypt_image_structure_preserving(image_array)
                    encrypted_array = np.frombuffer(encrypted_data[16:], dtype=np.uint8)
                    if len(encrypted_array) >= 32*32*3:
                        encrypted_array = encrypted_array[:32*32*3].reshape(32, 32, 3)
                        train_data_encrypted.append(encrypted_array)
                    else:
                        train_data_encrypted.append(np.zeros((32, 32, 3), dtype=np.uint8))
                else:
                    train_data_encrypted.append(np.zeros((32, 32, 3), dtype=np.uint8))
                
                if (i + 1) % 1000 == 0 or (i + 1) == train_size:
                    progress = (i + 1) / train_size * 0.5
                    self.progress_bar.set(progress)
                    self.log_message(f"AES-CBC encrypted {i+1}/{train_size} training images")
                    self.root.update_idletasks()

            self.log_message(f"Processing {test_size} test images with AES-CBC encryption...")
            for i in range(test_size):
                image_array, label = test_dataset_raw[i]
                test_data_original.append(image_array)
                test_labels.append(label)
                
                if self.encryption_key:
                    encrypted_data, _ = self.aes_processor.encrypt_image_structure_preserving(image_array)
                    encrypted_array = np.frombuffer(encrypted_data[16:], dtype=np.uint8)
                    if len(encrypted_array) >= 32*32*3:
                        encrypted_array = encrypted_array[:32*32*3].reshape(32, 32, 3)
                        test_data_encrypted.append(encrypted_array)
                    else:
                        test_data_encrypted.append(np.zeros((32, 32, 3), dtype=np.uint8))
                else:
                    test_data_encrypted.append(np.zeros((32, 32, 3), dtype=np.uint8))
                
                if (i + 1) % 500 == 0 or (i + 1) == test_size:
                    progress = 0.5 + (i + 1) / test_size * 0.5
                    self.progress_bar.set(progress)
                    self.log_message(f"AES-CBC encrypted {i+1}/{test_size} test images")
                    self.root.update_idletasks()

            self.train_dataset = HybridDataset(
                train_data_original, train_data_encrypted, train_labels, 
                encryption_metadata, mode=self.mode_var.get()
            )
            self.test_dataset = HybridDataset(
                test_data_original, test_data_encrypted, test_labels, 
                encryption_metadata, mode=self.mode_var.get()
            )

            if self.train_dataset.verify_integrity() and self.test_dataset.verify_integrity():
                self.log_message("Dataset integrity verified")
            else:
                self.log_message("Dataset integrity check failed!")

            self.progress_bar.set(1.0)
            self.log_message(f"AES-CBC dataset ready: {len(self.train_dataset)} training, {len(self.test_dataset)} test samples")
            self.status_label.configure(text="AES-CBC encrypted dataset loaded and ready")
            self.train_btn.configure(state="normal")
            if self.model:
                self.classify_btn.configure(state="normal")
            self.update_system_info()
            
        except Exception as e:
            self.log_message(f"Dataset loading failed: {str(e)}")
            messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")
            self.progress_bar.set(0)

    def start_training(self):
        if not hasattr(self, 'train_dataset') or self.train_dataset is None:
            messagebox.showerror("Error", "Please load dataset first!")
            return
        if self.is_training:
            messagebox.showwarning("Warning", "Training is already in progress!")
            return
            
        config = self.get_training_config()
        if not config:
            self.log_message("Training configuration cancelled.")
            return
            
        self.is_training = True
        self.train_btn.configure(text="Training...", state="disabled")
        self.training_thread = threading.Thread(target=self.train_model, args=(config,))
        self.training_thread.daemon = True
        self.training_thread.start()

    def get_training_config(self):
        config_window = ctk.CTkToplevel(self.root)
        config_window.title("Training Configuration")
        config_window.geometry("400x400")
        config_window.transient(self.root)
        config_window.grab_set()
        
        config = {}
        
        if self.mode_var.get() == "encrypted":
            labels_and_vars = [
                ("Number of Epochs:", "80", "epochs", int),
                ("Learning Rate:", "0.0015", "learning_rate", float),
                ("Batch Size:", "64", "batch_size", int),
                ("Dropout Rate (0.0-0.5):", "0.3", "dropout_rate", float),
            ]
        else:
            labels_and_vars = [
                ("Number of Epochs:", "50", "epochs", int),
                ("Learning Rate:", "0.001", "learning_rate", float),
                ("Batch Size:", "128", "batch_size", int),
                ("Dropout Rate (0.0-0.5):", "0.2", "dropout_rate", float),
            ]
        
        entries = {}
        for text, default_val, key, type_converter in labels_and_vars:
            ctk.CTkLabel(config_window, text=text).pack(pady=5)
            var = ctk.StringVar(value=default_val)
            entry = ctk.CTkEntry(config_window, textvariable=var)
            entry.pack(pady=2)
            entries[key] = (entry, var, type_converter)
        
        result = {"confirmed": False}
        
        def confirm():
            try:
                for key, (entry, var, type_converter) in entries.items():
                    config[key] = type_converter(var.get())
                result.update(config)
                result["confirmed"] = True
                config_window.destroy()
            except ValueError:
                messagebox.showerror("Input Error", "Please enter valid numeric values.")
        
        def cancel():
            config_window.destroy()
        
        button_frame = ctk.CTkFrame(config_window)
        button_frame.pack(pady=20)
        ctk.CTkButton(button_frame, text="Start Training", command=confirm).pack(side="left", padx=10)
        ctk.CTkButton(button_frame, text="Cancel", command=cancel).pack(side="left", padx=10)
        
        self.root.wait_window(config_window)
        return result if result["confirmed"] else None

    def train_model(self, config):
        try:
            self.log_message("Initializing optimized CNN training with AES-CBC encryption...")
            self.status_label.configure(text="Training in progress...")
            
            self.model = OptimizedCNN(
                num_classes=len(self.class_names),
                dropout_rate=config["dropout_rate"]
            ).to(self.device)
            
            self._update_info_labels()
            
            if self.mode_var.get() == "encrypted":
                criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
                optimizer = optim.AdamW(
                    self.model.parameters(), 
                    lr=config["learning_rate"],
                    weight_decay=5e-4,
                    betas=(0.9, 0.999),
                    eps=1e-7
                )
            else:
                criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
                optimizer = optim.AdamW(
                    self.model.parameters(), 
                    lr=config["learning_rate"], 
                    weight_decay=1e-4,
                    betas=(0.9, 0.999)
                )
            
            train_loader = DataLoader(
                self.train_dataset, 
                batch_size=config["batch_size"], 
                shuffle=True,
                num_workers=0,
                pin_memory=True if self.device.type == 'cuda' else False,
                drop_last=True
            )
            
            test_loader = DataLoader(
                self.test_dataset, 
                batch_size=config["batch_size"], 
                shuffle=False,
                num_workers=0,
                pin_memory=True if self.device.type == 'cuda' else False
            )
            
            if self.mode_var.get() == "encrypted":
                scheduler = optim.lr_scheduler.OneCycleLR(
                    optimizer, 
                    max_lr=config["learning_rate"] * 8,
                    steps_per_epoch=len(train_loader),
                    epochs=config["epochs"],
                    pct_start=0.4,
                    anneal_strategy='cos'
                )
            else:
                scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=10, T_mult=2, eta_min=1e-6
                )
            
            scaler = torch.amp.GradScaler() if self.device.type == 'cuda' else None
            
            self.training_metrics = {"losses": [], "accuracies": [], "val_accuracies": []}
            best_val_acc = 0.0
            patience = 15
            patience_counter = 0
            
            self.log_message(f"AES-CBC training configuration: {config}")
            self.log_message(f"Training mode: {self.mode_var.get()}")
            
            for epoch in range(config["epochs"]):
                if not self.is_training:
                    self.log_message("Training stopped by user.")
                    break
                
                start_time = time.time()
                
                self.model.train()
                running_loss, correct, total = 0.0, 0, 0
                
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                    
                    optimizer.zero_grad(set_to_none=True)
                    
                    use_amp = self.device.type == 'cuda'
                    with torch.amp.autocast(device_type=self.device.type, enabled=use_amp):
                        output = self.model(data)
                        loss = criterion(output, target)
                    
                    if scaler and use_amp:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                        optimizer.step()
                    
                    if self.mode_var.get() == "encrypted":
                        scheduler.step()
                    
                    running_loss += loss.item()
                    _, predicted = output.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()
                
                epoch_loss = running_loss / len(train_loader)
                epoch_acc = 100. * correct / total
                val_acc = self.evaluate_model(test_loader)
                
                if self.mode_var.get() == "original":
                    scheduler.step()
                
                self.training_metrics["losses"].append(epoch_loss)
                self.training_metrics["accuracies"].append(epoch_acc)
                self.training_metrics["val_accuracies"].append(val_acc)
                
                self.progress_bar.set((epoch + 1) / config["epochs"])
                epoch_time = time.time() - start_time
                
                self.log_message(
                    f"Epoch {epoch+1}/{config['epochs']}: "
                    f"Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, "
                    f"Val Acc: {val_acc:.2f}%, Time: {epoch_time:.1f}s, "
                    f"LR: {optimizer.param_groups[0]['lr']:.7f}"
                )
                
                self.update_metrics_display(epoch + 1, epoch_loss, epoch_acc, val_acc)
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    self.log_message(f"New best validation accuracy: {val_acc:.2f}% - Model saved!")
                    self.save_model(auto_save=True)
                else:
                    patience_counter += 1
                
                if patience_counter >= patience and epoch > 30:
                    self.log_message(f"Early stopping at epoch {epoch+1} - No improvement for {patience} epochs")
                    break
                
                if (epoch + 1) % 2 == 0 or epoch == 0:
                    self.root.after(10, self.plot_training_progress)
                
                self.root.update_idletasks()
            
            final_test_acc = self.evaluate_model(test_loader)
            self.log_message(
                f"AES-CBC training completed! Best Val Acc: {best_val_acc:.2f}%, "
                f"Final Test Acc: {final_test_acc:.2f}%"
            )
            self.root.after(10, self.plot_training_progress)
            
        except Exception as e:
            self.log_message(f"Training error: {str(e)}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Training Error", f"Training failed: {str(e)}")
        finally:
            self.is_training = False
            self.train_btn.configure(text="Train Model", state="normal")
            if hasattr(self, 'train_dataset') and self.train_dataset:
                self.classify_btn.configure(state="normal")
            self.status_label.configure(text="Training completed")
            self.progress_bar.set(0)

    def evaluate_model(self, data_loader):
        self.model.eval()
        correct, total = 0, 0
        
        with torch.no_grad():
            use_amp = self.device.type == 'cuda'
            with torch.amp.autocast(device_type=self.device.type, enabled=use_amp):
                for data, target in data_loader:
                    data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                    output = self.model(data)
                    _, predicted = output.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()
        
        return 100. * correct / total if total > 0 else 0

    def update_metrics_display(self, epoch, loss, train_acc, val_acc):
        metrics_text = (
            f"Epoch: {epoch}\n"
            f"Loss: {loss:.4f}\n"
            f"Train Acc: {train_acc:.2f}%\n"
            f"Val Acc: {val_acc:.2f}%\n"
            f"Device: {self.device}\n"
            f"Mode: {self.mode_var.get()}\n"
            f"Encryption: AES-CBC\n"
        )
        
        if hasattr(self, 'model') and self.model:
            total_params = sum(p.numel() for p in self.model.parameters())
            metrics_text += f"Parameters: {total_params:,}\n"
        
        self.metrics_text.delete("1.0", "end")
        self.metrics_text.insert("1.0", metrics_text)

    def plot_training_progress(self):
        if not self.training_metrics["losses"]:
            return
            
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.patch.set_facecolor('#2b2b2b')

        epochs = range(1, len(self.training_metrics["losses"]) + 1)

        ax1.plot(epochs, self.training_metrics["losses"], 'b-', linewidth=2.5, label='Training Loss')
        ax1.set_title('Training Loss (AES-CBC)', color='white', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', color='white', fontsize=12)
        ax1.set_ylabel('Loss', color='white', fontsize=12)
        ax1.tick_params(colors='white')
        ax1.set_facecolor('#3a3a3a')
        ax1.grid(True, alpha=0.3, color='gray', linestyle='--')
        ax1.legend(facecolor='#3a3a3a', edgecolor='white', labelcolor='white')

        ax2.plot(epochs, self.training_metrics["accuracies"], 'g-', linewidth=2.5, label='Training Acc')
        if self.training_metrics["val_accuracies"]:
            ax2.plot(epochs, self.training_metrics["val_accuracies"], 'r-', linewidth=2.5, label='Validation Acc')
        ax2.set_title('Accuracy (AES-CBC)', color='white', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch', color='white', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', color='white', fontsize=12)
        ax2.tick_params(colors='white')
        ax2.set_facecolor('#3a3a3a')
        ax2.legend(facecolor='#3a3a3a', edgecolor='white', labelcolor='white')
        ax2.grid(True, alpha=0.3, color='gray', linestyle='--')
        ax2.set_ylim(0, 100)

        plt.tight_layout(pad=3.0)

        canvas = FigureCanvasTkAgg(fig, self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def classify_image(self):
        if not self.model:
            messagebox.showerror("Error", "Please train or load a model first!")
            return
            
        file_path = filedialog.askopenfilename(
            title="Select Image for Classification", 
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        if not file_path:
            return
            
        try:
            self.log_message(f"Classifying image with AES-CBC: {Path(file_path).name}")
            
            image = Image.open(file_path).convert('RGB').resize((32, 32))
            image_array = np.array(image)
            
            if self.mode_var.get() == "encrypted" and self.encryption_key:
                encrypted_data, _ = self.aes_processor.encrypt_image_structure_preserving(image_array)
                encrypted_array = np.frombuffer(encrypted_data[16:], dtype=np.uint8)
                if len(encrypted_array) >= 32*32*3:
                    encrypted_array = encrypted_array[:32*32*3].reshape(32, 32, 3)
                    image_tensor = self.test_dataset.transform_encrypted(encrypted_array).unsqueeze(0).to(self.device)
                else:
                    image_tensor = torch.zeros(1, 3, 32, 32).to(self.device)
            else:
                image_tensor = self.test_dataset.transform_original(image_array).unsqueeze(0).to(self.device)
            
            self.model.eval()
            start_time = time.time()
            
            with torch.no_grad():
                use_amp = self.device.type == 'cuda'
                with torch.amp.autocast(device_type=self.device.type, enabled=use_amp):
                    output = self.model(image_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted_class_idx = torch.max(probabilities, 1)
                predicted_class = self.class_names[predicted_class_idx.item()]
            
            inference_time = (time.time() - start_time) * 1000
            
            top3_probs, top3_indices = torch.topk(probabilities, 3)
            result_text = f"AES-CBC Encrypted Classification Results:\n\n"
            for i in range(3):
                class_name = self.class_names[top3_indices[0][i].item()]
                prob = top3_probs[0][i].item()
                result_text += f"{i+1}. {class_name}: {prob:.2%}\n"
            
            result_text += f"\nInference Time: {inference_time:.1f}ms"
            result_text += f"\nEncryption: AES-CBC"
            
            self.log_message(
                f"AES-CBC Classification: {predicted_class} ({confidence.item():.2%}), "
                f"Time: {inference_time:.1f}ms"
            )
            messagebox.showinfo("AES-CBC Classification Result", result_text)
            
        except Exception as e:
            self.log_message(f"Classification error: {str(e)}")
            messagebox.showerror("Classification Error", f"Classification failed: {str(e)}")

    def save_model(self, auto_save=False):
        if not self.model:
            if not auto_save: 
                messagebox.showerror("Error", "No model to save!")
            return
            
        file_path = "best_aes_cbc_cnn_model.pth" if auto_save else filedialog.asksaveasfilename(
            title="Save Model", 
            defaultextension=".pth", 
            filetypes=[("PyTorch Model", "*.pth")]
        )
        
        if not file_path: 
            return
            
        try:
            model_data = {
                'model_state_dict': self.model.state_dict(),
                'model_architecture': 'OptimizedCNN',
                'num_classes': len(self.class_names),
                'encryption_mode': self.encryption_mode,
                'key_hash': self.key_hash,
                'training_metrics': self.training_metrics,
                'timestamp': time.time(),
                'version': '3.0-AES-CBC'
            }
            
            torch.save(model_data, file_path)
            
            if not auto_save:
                self.log_message(f"AES-CBC model saved successfully to {Path(file_path).name}")
                messagebox.showinfo("Success", f"AES-CBC Model saved successfully!\n\nFile: {Path(file_path).name}")
            else:
                self.log_message(f"AES-CBC Auto-save complete: {Path(file_path).name}")
                
        except Exception as e:
            log_msg = f"Save error: {str(e)}"
            self.log_message(log_msg)
            if not auto_save: 
                messagebox.showerror("Save Error", f"Failed to save model: {str(e)}")

    def load_model(self):
        file_path = filedialog.askopenfilename(
            title="Load Model", 
            filetypes=[("PyTorch Model", "*.pth")]
        )
        if not file_path: 
            return
            
        try:
            self.log_message(f"Loading AES-CBC model from {Path(file_path).name}...")
            self.status_label.configure(text="Loading AES-CBC model...")
            
            checkpoint = torch.load(file_path, map_location=self.device, weights_only=False)
            
            if not isinstance(checkpoint, dict) or 'model_state_dict' not in checkpoint:
                raise ValueError("Invalid checkpoint format.")
            
            self.model = OptimizedCNN(
                num_classes=checkpoint.get('num_classes', 10)
            ).to(self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            if 'training_metrics' in checkpoint:
                self.training_metrics = checkpoint['training_metrics']
                self.root.after(10, self.plot_training_progress)
            
            self._update_info_labels()
            
            if hasattr(self, 'train_dataset') and self.train_dataset:
                self.classify_btn.configure(state="normal")
            
            self.log_message(f"AES-CBC model loaded successfully from {Path(file_path).name}")
            self.status_label.configure(text="AES-CBC model loaded and ready")
            messagebox.showinfo("Success", f"AES-CBC Model loaded successfully!")
            
        except Exception as e:
            self.log_message(f"Load error: {str(e)}")
            messagebox.showerror("Load Error", f"Failed to load model: {str(e)}")
            self.model = None
            self._update_info_labels()

    def update_system_info(self):
        dataset_size = len(self.train_dataset) if self.train_dataset else 0
        info_text = (
            f"Dataset: {dataset_size} samples\n"
            f"Device: {self.device}\n"
            f"Architecture: Optimized CNN\n"
            f"Mode: {self.mode_var.get()}\n"
            f"Encryption: AES-CBC\n"
            f"Key Status: {'Active' if self.encryption_key else 'Inactive'}"
        )
        self.system_info.configure(text=info_text)

    def run(self):
        try:
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            self.root.mainloop()
        except Exception as e:
            self.log_message(f"Application error: {str(e)}")
        finally:
            if hasattr(self, 'is_training') and self.is_training:
                self.is_training = False

    def on_closing(self):
        if hasattr(self, 'is_training') and self.is_training:
            if messagebox.askokcancel("Quit", "Training is in progress. Do you want to quit?"):
                self.is_training = False
                if hasattr(self, 'training_thread') and self.training_thread:
                    self.training_thread.join(timeout=2.0)
                self.root.destroy()
        else:
            self.root.destroy()

class AdvancedAESProcessor:
    def __init__(self, key_size=32):
        self.key_size = key_size
        self.block_size = AES.block_size
        self.current_key = None

    def encrypt_image_format_preserving(self, image_array: np.ndarray) -> tuple:
        if self.current_key is None:
            raise ValueError("Encryption key is not set in AdvancedAESProcessor.")
        
        original_shape = image_array.shape
        height, width, channels = original_shape
        
        iv = get_random_bytes(AES.block_size)
        
        encrypted_channels = []
        
        for c in range(channels):
            channel = image_array[:, :, c].astype(np.uint8)
            
            cipher = AES.new(self.current_key, AES.MODE_CBC, iv)
            
            channel_bytes = channel.tobytes()
            padded_data = pad(channel_bytes, AES.block_size)
            encrypted_channel_bytes = cipher.encrypt(padded_data)
            
            encrypted_channel = np.frombuffer(encrypted_channel_bytes, dtype=np.uint8)
            
            target_size = height * width
            if len(encrypted_channel) > target_size:
                encrypted_channel = encrypted_channel[:target_size]
            elif len(encrypted_channel) < target_size:
                padding_needed = target_size - len(encrypted_channel)
                encrypted_channel = np.concatenate([
                    encrypted_channel, 
                    np.random.randint(0, 256, padding_needed, dtype=np.uint8)
                ])
            
            encrypted_channel = encrypted_channel.reshape(height, width)
            encrypted_channels.append(encrypted_channel)
        
        encrypted_image = np.stack(encrypted_channels, axis=2)
        return encrypted_image, iv

    def decrypt_image_format_preserving(self, encrypted_image: np.ndarray, iv: bytes, original_shape: tuple) -> np.ndarray:
        if self.current_key is None:
            raise ValueError("Decryption key is not set in AdvancedAESProcessor.")
        
        height, width, channels = original_shape
        decrypted_channels = []
        
        for c in range(channels):
            encrypted_channel = encrypted_image[:, :, c]
            
            cipher = AES.new(self.current_key, AES.MODE_CBC, iv)
            
            encrypted_channel_bytes = encrypted_channel.tobytes()
            
            original_channel_size = height * width
            padding_needed = (AES.block_size - (original_channel_size % AES.block_size)) % AES.block_size
            padded_size = original_channel_size + padding_needed
            
            if len(encrypted_channel_bytes) < padded_size:
                padding_bytes = padded_size - len(encrypted_channel_bytes)
                encrypted_channel_bytes += b'\x00' * padding_bytes
            elif len(encrypted_channel_bytes) > padded_size:
                encrypted_channel_bytes = encrypted_channel_bytes[:padded_size]
            
            try:
                decrypted_padded = cipher.decrypt(encrypted_channel_bytes)
                decrypted_channel_bytes = unpad(decrypted_padded, AES.block_size)
            except ValueError:
                decrypted_channel_bytes = decrypted_padded[:original_channel_size]
            
            if len(decrypted_channel_bytes) > original_channel_size:
                decrypted_channel_bytes = decrypted_channel_bytes[:original_channel_size]
            elif len(decrypted_channel_bytes) < original_channel_size:
                padding_needed = original_channel_size - len(decrypted_channel_bytes)
                decrypted_channel_bytes += b'\x00' * padding_needed
            
            decrypted_channel = np.frombuffer(decrypted_channel_bytes, dtype=np.uint8).reshape(height, width)
            decrypted_channels.append(decrypted_channel)
        
        decrypted_image = np.stack(decrypted_channels, axis=2)
        return decrypted_image

    def encrypt_image_structure_preserving(self, image_array: np.ndarray) -> tuple:
        return self.encrypt_image_format_preserving(image_array)

    def decrypt_image_structure_preserving(self, encrypted_data: bytes, original_shape: tuple) -> np.ndarray:
        iv = encrypted_data[:AES.block_size]
        encrypted_bytes = encrypted_data[AES.block_size:]
        encrypted_image = np.frombuffer(encrypted_bytes, dtype=np.uint8).reshape(original_shape)
        return self.decrypt_image_format_preserving(encrypted_image, iv, original_shape)

    def encrypt_image_cbc(self, image_array: np.ndarray) -> tuple:
        if self.current_key is None:
            raise ValueError("Encryption key is not set in AdvancedAESProcessor.")
        
        original_shape = image_array.shape
        
        iv = get_random_bytes(AES.block_size)
        cipher = AES.new(self.current_key, AES.MODE_CBC, iv)
        
        image_bytes = image_array.astype(np.uint8).tobytes()
        padded_data = pad(image_bytes, AES.block_size)
        encrypted_data = cipher.encrypt(padded_data)
        
        return iv + encrypted_data, iv

    def decrypt_image_cbc(self, encrypted_data: bytes, original_shape: tuple) -> np.ndarray:
        if self.current_key is None:
            raise ValueError("Decryption key is not set in AdvancedAESProcessor.")
        
        iv = encrypted_data[:AES.block_size]
        ciphertext = encrypted_data[AES.block_size:]
        
        cipher = AES.new(self.current_key, AES.MODE_CBC, iv)
        decrypted_padded = cipher.decrypt(ciphertext)
        
        try:
            decrypted_data = unpad(decrypted_padded, AES.block_size)
        except ValueError:
            expected_size = np.prod(original_shape)
            decrypted_data = decrypted_padded[:expected_size]
        
        expected_size = np.prod(original_shape)
        if len(decrypted_data) > expected_size:
            decrypted_data = decrypted_data[:expected_size]
        elif len(decrypted_data) < expected_size:
            padding_needed = expected_size - len(decrypted_data)
            decrypted_data += b'\x00' * padding_needed
        
        decrypted_image = np.frombuffer(decrypted_data, dtype=np.uint8).reshape(original_shape)
        return decrypted_image

    def set_key(self, key: bytes):
        if len(key) != self.key_size:
            raise ValueError(f"Key must be {self.key_size} bytes long")
        self.current_key = key

    def encrypt_with_metadata(self, image_array: np.ndarray) -> dict:
        if self.current_key is None:
            raise ValueError("Encryption key is not set")
        
        encrypted_image, iv = self.encrypt_image_format_preserving(image_array)
        
        metadata = {
            'encrypted_data': encrypted_image.tobytes(),
            'iv': iv,
            'shape': image_array.shape,
            'dtype': str(image_array.dtype),
            'encryption_mode': 'AES-CBC-FPE',
            'timestamp': time.time()
        }
        
        return metadata

    def decrypt_with_metadata(self, metadata: dict) -> np.ndarray:
        if self.current_key is None:
            raise ValueError("Decryption key is not set")
        
        encrypted_data = metadata['encrypted_data']
        iv = metadata['iv']
        shape = metadata['shape']
        
        encrypted_image = np.frombuffer(encrypted_data, dtype=np.uint8).reshape(shape)
        decrypted_image = self.decrypt_image_format_preserving(encrypted_image, iv, shape)
        
        return decrypted_image

def load_and_preprocess_dataset(dataset_size_limit=None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 255).byte().numpy().transpose(1, 2, 0))
    ])
    
    train_dataset_raw = CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset_raw = CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    if dataset_size_limit:
        train_size = min(dataset_size_limit, len(train_dataset_raw))
        test_size = min(dataset_size_limit // 5, len(test_dataset_raw))
    else:
        train_size = len(train_dataset_raw)
        test_size = len(test_dataset_raw)
    
    return train_dataset_raw, test_dataset_raw, train_size, test_size

def verify_encryption_decryption(aes_processor, sample_image):
    try:
        original_shape = sample_image.shape
        
        encrypted_image, iv = aes_processor.encrypt_image_format_preserving(sample_image)
        
        encrypted_bytes = iv + encrypted_image.tobytes()
        decrypted_image = aes_processor.decrypt_image_structure_preserving(encrypted_bytes, original_shape)
        
        mse = np.mean((sample_image.astype(float) - decrypted_image.astype(float)) ** 2)
        
        return mse < 1e-6, mse, encrypted_image.shape == original_shape
    except Exception as e:
        return False, float('inf'), False

def enhanced_data_augmentation():
    return {
        'original': transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'encrypted': transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(degrees=5),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    }

class ImprovedHybridDataset(HybridDataset):
    def __init__(self, original_data, encrypted_data, labels, encryption_metadata, mode='original'):
        super().__init__(original_data, encrypted_data, labels, encryption_metadata, mode)
        
        augmentations = enhanced_data_augmentation()
        self.transform_original = augmentations['original']
        self.transform_encrypted = augmentations['encrypted']
        
        self.encryption_verification_passed = self._verify_encryption_integrity()

    def _verify_encryption_integrity(self):
        if len(self.encrypted_data) == 0:
            return False
        
        try:
            sample_encrypted = self.encrypted_data[0]
            if isinstance(sample_encrypted, np.ndarray):
                return sample_encrypted.shape == self.original_data[0].shape
            return True
        except:
            return False

    def get_sample_for_visualization(self, idx):
        if idx >= len(self.original_data):
            return None, None
        
        original = self.original_data[idx]
        encrypted = self.encrypted_data[idx] if idx < len(self.encrypted_data) else None
        
        return original, encrypted

def create_advanced_model_architecture(num_classes=10, dropout_rate=0.3):
    class AdvancedOptimizedCNN(nn.Module):
        def __init__(self, num_classes, dropout_rate):
            super(AdvancedOptimizedCNN, self).__init__()
            
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(dropout_rate * 0.2),
                
                nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(dropout_rate * 0.3),
                
                nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(dropout_rate * 0.4),
                
                nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((2, 2)),
            )
            
            self.classifier = nn.Sequential(
                nn.Linear(512 * 2 * 2, 1024, bias=False),
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(1024, 512, bias=False),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate * 0.7),
                nn.Linear(512, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate * 0.5),
                nn.Linear(256, num_classes)
            )
            
            self._initialize_weights()

        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        def forward(self, x):
            x = self.features(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x
    
    return AdvancedOptimizedCNN(num_classes, dropout_rate)

def get_optimized_training_config(mode='original'):
    if mode == 'encrypted':
        return {
            'epochs': 100,
            'learning_rate': 0.002,
            'batch_size': 64,
            'dropout_rate': 0.4,
            'weight_decay': 1e-3,
            'label_smoothing': 0.15,
            'patience': 20
        }
    else:
        return {
            'epochs': 75,
            'learning_rate': 0.001,
            'batch_size': 128,
            'dropout_rate': 0.25,
            'weight_decay': 5e-4,
            'label_smoothing': 0.1,
            'patience': 15
        }

if __name__ == "__main__":
    try:
        app = SecureImageClassifierApp()
        app.run()
    except Exception as e:
        print(f"Application startup error: {str(e)}")
        import traceback
        traceback.print_exc()