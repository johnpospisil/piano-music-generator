"""
Music tokenizer for converting events to/from integer tokens.
"""

from typing import List, Dict, Tuple, Optional
import json
from pathlib import Path


class MusicTokenizer:
    """
    Tokenizer for music events.
    Converts musical events into integer tokens for neural network training.
    """
    
    def __init__(
        self,
        num_pitches: int = 88,  # Standard piano range (21-108)
        velocity_bins: int = 32,
        duration_bins: int = 64,
        time_shift_bins: int = 100,
        min_pitch: int = 21,
    ):
        """
        Initialize tokenizer.
        
        Args:
            num_pitches: Number of unique pitches
            velocity_bins: Number of velocity bins
            duration_bins: Number of duration bins
            time_shift_bins: Number of time shift bins
            min_pitch: Minimum MIDI pitch
        """
        self.num_pitches = num_pitches
        self.velocity_bins = velocity_bins
        self.duration_bins = duration_bins
        self.time_shift_bins = time_shift_bins
        self.min_pitch = min_pitch
        
        # Build vocabulary
        self.vocab = self._build_vocabulary()
        self.vocab_size = len(self.vocab)
        
        # Create reverse mapping
        self.idx_to_token = {idx: token for token, idx in self.vocab.items()}
        
        # Special tokens
        self.pad_token = '<PAD>'
        self.sos_token = '<SOS>'  # Start of sequence
        self.eos_token = '<EOS>'  # End of sequence
        self.unk_token = '<UNK>'  # Unknown
        
        self.pad_idx = self.vocab[self.pad_token]
        self.sos_idx = self.vocab[self.sos_token]
        self.eos_idx = self.vocab[self.eos_token]
        self.unk_idx = self.vocab[self.unk_token]
    
    def _build_vocabulary(self) -> Dict[str, int]:
        """Build token vocabulary."""
        vocab = {}
        idx = 0
        
        # Special tokens
        vocab['<PAD>'] = idx
        idx += 1
        vocab['<SOS>'] = idx
        idx += 1
        vocab['<EOS>'] = idx
        idx += 1
        vocab['<UNK>'] = idx
        idx += 1
        
        # Note-on tokens (pitch)
        for pitch in range(self.num_pitches):
            vocab[f'NOTE_{pitch}'] = idx
            idx += 1
        
        # Velocity tokens
        for vel in range(self.velocity_bins):
            vocab[f'VEL_{vel}'] = idx
            idx += 1
        
        # Duration tokens
        for dur in range(self.duration_bins):
            vocab[f'DUR_{dur}'] = idx
            idx += 1
        
        # Time shift tokens
        for ts in range(self.time_shift_bins):
            vocab[f'TIME_{ts}'] = idx
            idx += 1
        
        return vocab
    
    def encode_event(self, event: Dict) -> List[int]:
        """
        Encode a single event to token indices.
        
        Args:
            event: Event dictionary with type and values
            
        Returns:
            List of token indices
        """
        tokens = []
        
        if event['type'] == 'note':
            # For a note: [NOTE_pitch, VEL_velocity, DUR_duration]
            pitch_offset = event['pitch'] - self.min_pitch
            if 0 <= pitch_offset < self.num_pitches:
                tokens.append(self.vocab[f'NOTE_{pitch_offset}'])
                tokens.append(self.vocab[f'VEL_{event["velocity"]}'])
                tokens.append(self.vocab[f'DUR_{event["duration"]}'])
        
        elif event['type'] == 'time_shift':
            # For time shift: [TIME_value]
            tokens.append(self.vocab[f'TIME_{event["value"]}'])
        
        return tokens
    
    def decode_tokens(self, tokens: List[int]) -> List[Dict]:
        """
        Decode token indices back to events.
        
        Args:
            tokens: List of token indices
            
        Returns:
            List of event dictionaries
        """
        events = []
        i = 0
        
        while i < len(tokens):
            token_str = self.idx_to_token.get(tokens[i], '<UNK>')
            
            # Skip special tokens
            if token_str in ['<PAD>', '<SOS>', '<EOS>', '<UNK>']:
                i += 1
                continue
            
            if token_str.startswith('NOTE_'):
                # Expect NOTE, VEL, DUR sequence
                if i + 2 < len(tokens):
                    pitch_offset = int(token_str.split('_')[1])
                    vel_token = self.idx_to_token.get(tokens[i + 1], '')
                    dur_token = self.idx_to_token.get(tokens[i + 2], '')
                    
                    if vel_token.startswith('VEL_') and dur_token.startswith('DUR_'):
                        velocity = int(vel_token.split('_')[1])
                        duration = int(dur_token.split('_')[1])
                        
                        events.append({
                            'type': 'note',
                            'pitch': pitch_offset + self.min_pitch,
                            'velocity': velocity,
                            'duration': duration
                        })
                        i += 3
                        continue
                i += 1
            
            elif token_str.startswith('TIME_'):
                time_shift = int(token_str.split('_')[1])
                events.append({
                    'type': 'time_shift',
                    'value': time_shift
                })
                i += 1
            
            else:
                i += 1
        
        return events
    
    def encode_sequence(self, events: List[Dict], add_special_tokens: bool = True) -> List[int]:
        """
        Encode a sequence of events to tokens.
        
        Args:
            events: List of event dictionaries
            add_special_tokens: Whether to add SOS/EOS tokens
            
        Returns:
            List of token indices
        """
        tokens = []
        
        if add_special_tokens:
            tokens.append(self.sos_idx)
        
        for event in events:
            tokens.extend(self.encode_event(event))
        
        if add_special_tokens:
            tokens.append(self.eos_idx)
        
        return tokens
    
    def save_vocab(self, path: Path):
        """Save vocabulary to JSON file."""
        vocab_dict = {
            'vocab': self.vocab,
            'config': {
                'num_pitches': self.num_pitches,
                'velocity_bins': self.velocity_bins,
                'duration_bins': self.duration_bins,
                'time_shift_bins': self.time_shift_bins,
                'min_pitch': self.min_pitch,
            }
        }
        with open(path, 'w') as f:
            json.dump(vocab_dict, f, indent=2)
    
    @classmethod
    def load_vocab(cls, path: Path) -> 'MusicTokenizer':
        """Load vocabulary from JSON file."""
        with open(path, 'r') as f:
            vocab_dict = json.load(f)
        
        config = vocab_dict['config']
        return cls(
            num_pitches=config['num_pitches'],
            velocity_bins=config['velocity_bins'],
            duration_bins=config['duration_bins'],
            time_shift_bins=config['time_shift_bins'],
            min_pitch=config['min_pitch'],
        )
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size
