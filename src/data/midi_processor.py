"""
MIDI file processing utilities.
Converts MIDI files into structured note sequences.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import mido
from mido import MidiFile, MidiTrack, Message
import pretty_midi
from dataclasses import dataclass


@dataclass
class Note:
    """Represents a musical note with timing information."""
    pitch: int  # MIDI note number (0-127)
    velocity: int  # Note velocity (0-127)
    start_time: float  # Start time in seconds
    end_time: float  # End time in seconds
    
    @property
    def duration(self) -> float:
        """Get note duration in seconds."""
        return self.end_time - self.start_time


class MidiProcessor:
    """
    Processes MIDI files and extracts note sequences.
    """
    
    def __init__(
        self,
        velocity_bins: int = 32,
        duration_bins: int = 64,
        time_shift_bins: int = 100,
        min_pitch: int = 21,  # A0 - lowest piano note
        max_pitch: int = 108,  # C8 - highest piano note
    ):
        """
        Initialize MIDI processor.
        
        Args:
            velocity_bins: Number of bins for velocity quantization
            duration_bins: Number of bins for duration quantization
            time_shift_bins: Number of bins for time shift quantization
            min_pitch: Minimum MIDI pitch to consider
            max_pitch: Maximum MIDI pitch to consider
        """
        self.velocity_bins = velocity_bins
        self.duration_bins = duration_bins
        self.time_shift_bins = time_shift_bins
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch
        
        # Define quantization ranges
        self.max_duration = 4.0  # Max duration in seconds (whole note at 60 BPM)
        self.max_time_shift = 2.0  # Max time shift in seconds
        
    def load_midi(self, midi_path: Path) -> List[Note]:
        """
        Load MIDI file and extract notes.
        
        Args:
            midi_path: Path to MIDI file
            
        Returns:
            List of Note objects sorted by start time
        """
        try:
            pm = pretty_midi.PrettyMIDI(str(midi_path))
            notes = []
            
            # Extract notes from all instruments (usually just piano)
            for instrument in pm.instruments:
                if instrument.is_drum:
                    continue
                    
                for note in instrument.notes:
                    if self.min_pitch <= note.pitch <= self.max_pitch:
                        notes.append(Note(
                            pitch=note.pitch,
                            velocity=note.velocity,
                            start_time=note.start,
                            end_time=note.end
                        ))
            
            # Sort by start time
            notes.sort(key=lambda x: x.start_time)
            return notes
            
        except Exception as e:
            raise ValueError(f"Error loading MIDI file {midi_path}: {str(e)}")
    
    def quantize_velocity(self, velocity: int) -> int:
        """Quantize velocity to bins."""
        return int(velocity / 127 * (self.velocity_bins - 1))
    
    def quantize_duration(self, duration: float) -> int:
        """Quantize duration to bins."""
        duration = min(duration, self.max_duration)
        return int(duration / self.max_duration * (self.duration_bins - 1))
    
    def quantize_time_shift(self, time_shift: float) -> int:
        """Quantize time shift to bins."""
        time_shift = min(time_shift, self.max_time_shift)
        return int(time_shift / self.max_time_shift * (self.time_shift_bins - 1))
    
    def dequantize_velocity(self, velocity_bin: int) -> int:
        """Convert velocity bin back to MIDI velocity."""
        return int(velocity_bin / (self.velocity_bins - 1) * 127)
    
    def dequantize_duration(self, duration_bin: int) -> float:
        """Convert duration bin back to seconds."""
        return (duration_bin / (self.duration_bins - 1)) * self.max_duration
    
    def dequantize_time_shift(self, time_shift_bin: int) -> float:
        """Convert time shift bin back to seconds."""
        return (time_shift_bin / (self.time_shift_bins - 1)) * self.max_time_shift
    
    def notes_to_events(self, notes: List[Note]) -> List[Dict]:
        """
        Convert notes to event sequence.
        Each event contains: type, pitch, velocity, duration, time_shift
        
        Args:
            notes: List of Note objects
            
        Returns:
            List of event dictionaries
        """
        events = []
        current_time = 0.0
        
        for note in notes:
            # Add time shift event if needed
            time_shift = note.start_time - current_time
            if time_shift > 0:
                events.append({
                    'type': 'time_shift',
                    'value': self.quantize_time_shift(time_shift),
                    'time': current_time
                })
                current_time = note.start_time
            
            # Add note event
            events.append({
                'type': 'note',
                'pitch': note.pitch,
                'velocity': self.quantize_velocity(note.velocity),
                'duration': self.quantize_duration(note.duration),
                'time': current_time
            })
            
        return events
    
    def events_to_notes(self, events: List[Dict]) -> List[Note]:
        """
        Convert event sequence back to notes.
        
        Args:
            events: List of event dictionaries
            
        Returns:
            List of Note objects
        """
        notes = []
        current_time = 0.0
        
        for event in events:
            if event['type'] == 'time_shift':
                current_time += self.dequantize_time_shift(event['value'])
            elif event['type'] == 'note':
                duration = self.dequantize_duration(event['duration'])
                notes.append(Note(
                    pitch=event['pitch'],
                    velocity=self.dequantize_velocity(event['velocity']),
                    start_time=current_time,
                    end_time=current_time + duration
                ))
        
        return notes
    
    def save_midi(self, notes: List[Note], output_path: Path, tempo: int = 120):
        """
        Save notes to MIDI file.
        
        Args:
            notes: List of Note objects
            output_path: Output path for MIDI file
            tempo: Tempo in BPM
        """
        # Create MIDI file
        pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano
        
        # Add notes
        for note in notes:
            midi_note = pretty_midi.Note(
                velocity=note.velocity,
                pitch=note.pitch,
                start=note.start_time,
                end=note.end_time
            )
            instrument.notes.append(midi_note)
        
        pm.instruments.append(instrument)
        pm.write(str(output_path))
    
    def extract_features(self, notes: List[Note]) -> Dict[str, np.ndarray]:
        """
        Extract statistical features from note sequence.
        
        Args:
            notes: List of Note objects
            
        Returns:
            Dictionary of feature arrays
        """
        if not notes:
            return {
                'pitches': np.array([]),
                'velocities': np.array([]),
                'durations': np.array([]),
                'intervals': np.array([])
            }
        
        pitches = np.array([n.pitch for n in notes])
        velocities = np.array([n.velocity for n in notes])
        durations = np.array([n.duration for n in notes])
        
        # Calculate pitch intervals
        intervals = np.diff(pitches) if len(pitches) > 1 else np.array([])
        
        return {
            'pitches': pitches,
            'velocities': velocities,
            'durations': durations,
            'intervals': intervals
        }
