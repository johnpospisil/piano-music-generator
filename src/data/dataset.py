"""
PyTorch Dataset classes for MIDI music generation.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
from .midi_processor import MidiProcessor
from .tokenizer import MusicTokenizer


class MidiDataset(Dataset):
    """
    PyTorch Dataset for MIDI files.
    """
    
    def __init__(
        self,
        metadata_df: pd.DataFrame,
        data_root: Path,
        processor: MidiProcessor,
        tokenizer: MusicTokenizer,
        sequence_length: int = 512,
        stride: int = 256,
        split: str = 'train',
    ):
        """
        Initialize MIDI dataset.
        
        Args:
            metadata_df: DataFrame with MIDI file metadata
            data_root: Root directory containing MIDI files
            processor: MidiProcessor instance
            tokenizer: MusicTokenizer instance
            sequence_length: Length of token sequences
            stride: Stride for creating overlapping sequences
            split: Dataset split ('train', 'validation', or 'test')
        """
        self.metadata_df = metadata_df[metadata_df['split'] == split].reset_index(drop=True)
        self.data_root = Path(data_root)
        self.processor = processor
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.stride = stride
        self.split = split
        
        # Store processed sequences
        self.sequences = []
        self.file_indices = []  # Track which file each sequence came from
        
        print(f"Processing {len(self.metadata_df)} {split} files...")
        self._preprocess_files()
        print(f"Created {len(self.sequences)} sequences from {split} set")
    
    def _preprocess_files(self):
        """Preprocess all MIDI files and create sequences."""
        for idx, row in self.metadata_df.iterrows():
            try:
                midi_path = self.data_root / row['midi_filename']
                
                if not midi_path.exists():
                    continue
                
                # Load and process MIDI file
                notes = self.processor.load_midi(midi_path)
                
                if not notes:
                    continue
                
                # Convert to events
                events = self.processor.notes_to_events(notes)
                
                # Encode to tokens
                tokens = self.tokenizer.encode_sequence(events, add_special_tokens=False)
                
                # Create overlapping sequences
                for i in range(0, len(tokens) - self.sequence_length, self.stride):
                    seq = tokens[i:i + self.sequence_length]
                    if len(seq) == self.sequence_length:
                        self.sequences.append(seq)
                        self.file_indices.append(idx)
                
                # Print progress every 100 files
                if (idx + 1) % 100 == 0:
                    print(f"  Processed {idx + 1}/{len(self.metadata_df)} files, "
                          f"{len(self.sequences)} sequences created")
            
            except Exception as e:
                print(f"  Error processing {row['midi_filename']}: {str(e)}")
                continue
    
    def __len__(self) -> int:
        """Return number of sequences."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a training example.
        
        Args:
            idx: Sequence index
            
        Returns:
            Tuple of (input_sequence, target_sequence)
            Input is shifted by 1 position for next-token prediction
        """
        sequence = self.sequences[idx]
        
        # Input: all tokens except the last
        input_seq = torch.tensor(sequence[:-1], dtype=torch.long)
        
        # Target: all tokens except the first
        target_seq = torch.tensor(sequence[1:], dtype=torch.long)
        
        return input_seq, target_seq
    
    def get_metadata(self, idx: int) -> pd.Series:
        """Get metadata for the file that generated this sequence."""
        file_idx = self.file_indices[idx]
        return self.metadata_df.iloc[file_idx]


class InferenceDataset(Dataset):
    """
    Dataset for music generation/inference.
    Takes a seed sequence and prepares it for generation.
    """
    
    def __init__(
        self,
        seed_tokens: List[int],
        sequence_length: int = 512,
    ):
        """
        Initialize inference dataset.
        
        Args:
            seed_tokens: Initial token sequence
            sequence_length: Maximum sequence length
        """
        self.seed_tokens = seed_tokens[-sequence_length:]  # Take last N tokens
        self.sequence_length = sequence_length
    
    def __len__(self) -> int:
        return 1
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Return seed sequence as tensor."""
        # Pad if necessary
        if len(self.seed_tokens) < self.sequence_length:
            padding = [0] * (self.sequence_length - len(self.seed_tokens))
            tokens = padding + self.seed_tokens
        else:
            tokens = self.seed_tokens
        
        return torch.tensor(tokens, dtype=torch.long)


def create_dataloaders(
    metadata_path: Path,
    data_root: Path,
    processor: MidiProcessor,
    tokenizer: MusicTokenizer,
    batch_size: int = 32,
    sequence_length: int = 512,
    stride: int = 256,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        metadata_path: Path to metadata CSV
        data_root: Root directory containing MIDI files
        processor: MidiProcessor instance
        tokenizer: MusicTokenizer instance
        batch_size: Batch size for training
        sequence_length: Length of token sequences
        stride: Stride for overlapping sequences
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Load metadata
    metadata = pd.read_csv(metadata_path)
    
    # Create datasets
    train_dataset = MidiDataset(
        metadata, data_root, processor, tokenizer,
        sequence_length, stride, split='train'
    )
    
    val_dataset = MidiDataset(
        metadata, data_root, processor, tokenizer,
        sequence_length, stride, split='validation'
    )
    
    test_dataset = MidiDataset(
        metadata, data_root, processor, tokenizer,
        sequence_length, stride, split='test'
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
