import os
import torch
import torch.nn as nn
import pretty_midi
from mido import MidiFile, MidiTrack, Message

# Optional: Uncomment if you want to convert MIDI -> WAV
# from midi2audio import FluidSynth

# --- Step 1: Load MIDI files and extract note pitch sequences ---
def load_midi_files(folder="midi_songs"):
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Folder '{folder}' created. Please add MIDI files (.mid) there and rerun.")
        return []

    sequences = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.mid', '.midi')):
            try:
                midi_data = pretty_midi.PrettyMIDI(os.path.join(folder, filename))
                notes = []
                for instrument in midi_data.instruments:
                    if not instrument.is_drum:
                        for note in instrument.notes:
                            notes.append(note.pitch)
                if notes:
                    sequences.append(notes[:100])  # truncate for uniform length
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    print(f"Loaded {len(sequences)} sequences from {len(os.listdir(folder))} MIDI files.")
    return sequences

# --- Step 2: Convert sequences to tensors ---
def sequences_to_tensor(sequences):
    tensor_data = []
    for seq in sequences:
        tensor_data.append(torch.tensor(seq, dtype=torch.float32).unsqueeze(1))  # (seq_len, 1)
    return tensor_data

# --- Step 3: Define LSTM Generator model ---
class LSTMGenerator(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, output_size=1):
        super(LSTMGenerator, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        out, hidden = self.lstm(x, hidden)
        out = self.linear(out)
        return out, hidden

# --- Step 4: Save generated notes as a MIDI file ---
def save_midi(note_sequence, filename="generated_music.mid"):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    for note in note_sequence:
        note_val = int(max(0, min(127, note)))
        track.append(Message('note_on', note=note_val, velocity=64, time=200))
        track.append(Message('note_off', note=note_val, velocity=64, time=300))

    mid.save(filename)
    print(f"Saved generated music to {filename}")

# --- Optional Step 5: Convert MIDI to WAV audio ---
# def midi_to_wav(midi_path="generated_music.mid", wav_path="output.wav", soundfont_path="path_to_soundfont.sf2"):
#     fs = FluidSynth(sound_font=soundfont_path)
#     fs.midi_to_audio(midi_path, wav_path)
#     print(f"Converted {midi_path} to {wav_path}")

# --- Step 6: Train LSTM and generate music ---
def train_generator():
    sequences = load_midi_files()
    if not sequences:
        print("No MIDI sequences found. Add MIDI files to 'midi_songs' folder and rerun.")
        return

    tensor_data = sequences_to_tensor(sequences)

    input_size = 1
    hidden_size = 128
    num_layers = 2
    output_size = 1
    seq_len = 100

    model = LSTMGenerator(input_size, hidden_size, num_layers, output_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    epochs = 200
    print("Starting training...")
    for epoch in range(epochs):
        total_loss = 0
        for seq in tensor_data:
            if seq.size(0) < seq_len:
                padding = torch.zeros(seq_len - seq.size(0), 1)
                seq = torch.cat([seq, padding], dim=0)
            else:
                seq = seq[:seq_len]

            seq = seq.unsqueeze(0)  # batch_size=1

            input_seq = seq[:, :-1, :]
            target_seq = seq[:, 1:, :]

            optimizer.zero_grad()
            output, _ = model(input_seq)
            loss = criterion(output, target_seq)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(tensor_data):.4f}")

    print("Training complete! Generating music...")

    model.eval()
    generated = [60.0]  # start with Middle C note
    input_seq = torch.tensor([[[60.0]]])  # shape (1,1,1)
    hidden = None

    with torch.no_grad():
        for _ in range(seq_len - 1):
            output, hidden = model(input_seq, hidden)
            note = output[:, -1, :].item()
            generated.append(note)
            input_seq = torch.tensor([[[note]]])

    save_midi(generated, "generated_music.mid")

    # --- Optional: Convert to WAV (uncomment below and provide your soundfont path) ---
    # midi_to_wav("generated_music.mid", "generated_music.wav", "C:/soundfonts/FluidR3_GM.sf2")

if __name__ == "__main__":
    train_generator()
