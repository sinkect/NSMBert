import argparse
import csv
import io
import os
import random

from datasets import load_dataset, Audio
from pydub import AudioSegment


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process music dataset and split into segments.')
    parser.add_argument('--num-songs', type=int, default=25000, 
                        help='Number of songs to process (default: 25000)')
    parser.add_argument('--batch-size', type=int, default=100, 
                        help='Number of songs to process in each batch (default: 100)')
    parser.add_argument('--output-dir', type=str, default='music_split', 
                        help='Directory to save the output files (default: music_split)')
    parser.add_argument('--csv-file', type=str, default='processed_music.csv', 
                        help='CSV file to store metadata (default: processed_music.csv)')
    parser.add_argument('--resume', type=int,default=0,
                        help='Resume processing from the last processed song')
    parser.add_argument('--test', action='store_true', 
                        help='Run in test mode with only 5 songs')
    parser.add_argument('--selection-threshold', type=float, default=0.5, 
                        help='Threshold for random selection (default: 0.5, higher values select more songs)')
    return parser.parse_args()

def main():
    # Parse command line arguments
    global ds_iter
    args = parse_arguments()

    # If test mode is enabled, override num_songs
    if args.test:
        args.num_songs = 5
        print("Running in test mode with 5 songs")

    # Create output directory if it doesn't exist
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # CSV file to store metadata
    csv_file = args.csv_file
    csv_header = ["title", "artist", "first_15s_path", "last_15s_path"]

    # Check if we should resume processing
    processed_count = 0
    processed_songs = set()  # Store (title, artist) tuples

    # Load the dataset from Hugging Face
    print("Loading dataset...")
    ds = load_dataset(
        "benjamin-paine/free-music-archive-full",
        split="train",
        streaming=True
    )
    ds = ds.cast_column("audio", Audio(decode=False))

    if args.resume > 0  and os.path.exists(csv_file):
        # Read existing CSV to get processed songs
        with open(csv_file, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 4:  # title, artist, first_15s_path, last_15s_path
                    processed_songs.add((row[0], row[1]))  # (title, artist)
                    processed_count += 1

            resume_seen = args.resume  # 파일이나 로그에서 읽어와야 함

            ds_iter = iter(ds)
            for i in range(resume_seen):
                try:
                    print(i)
                    next(ds_iter)
                except StopIteration:
                    ds_iter = iter(ds)  # 다시 처음부터 (loop dataset)
                    next(ds_iter)

        print(f"Resuming from {processed_count} previously processed songs")
    else:
        # Create new CSV file
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)


        # Convert to iterator
        ds_iter = iter(ds)

    # Number of songs to process
    num_songs = args.num_songs
    batch_size = args.batch_size

    print(f"Processing {num_songs} songs...")

    # Process songs
    # Keep track of how many songs we've seen (for reporting purposes)
    songs_seen = 0
    # Get the selection threshold from arguments
    selection_threshold = args.selection_threshold

    while processed_count < num_songs:
        try:
            # Get next song
            song = next(ds_iter)
            songs_seen += 1

            # Random selection: only process songs that pass the random threshold
            # Higher threshold = more songs selected
            if random.random() > selection_threshold:
                continue

            # Extract audio data, title, and artist
            audio_data = song["audio"]
            title = song["title"]
            artist = song["artist"] if "artist" in song else "Unknown Artist"

            # Skip if title is empty or None
            if not title:
                continue

            # Skip if we've already processed this title and artist combination (duplicate detection)
            if (title, artist) in processed_songs:
                print(f"Duplicate song '{title}' by '{artist}' found, skipping...")
                continue

            # Clean title and artist for filename
            clean_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in title)
            clean_artist = "".join(c if c.isalnum() or c in " -_" else "_" for c in artist)

            try:
                # The audio data is a dictionary with keys 'bytes' and 'path'
                # We'll use the 'bytes' key to load the audio directly

                # Check if audio_data is a dictionary with required keys
                if not (isinstance(audio_data, dict) and 'bytes' in audio_data):
                    print(f"Audio data for song '{title}' doesn't have the expected structure, skipping...")
                    continue

                # Get the audio bytes
                audio_bytes = audio_data['bytes']

                # Load audio directly from bytes
                audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")

                # Get audio duration in milliseconds
                duration_ms = len(audio)

                # Skip if audio is too short
                if duration_ms < 30000:  # 30 seconds
                    print(f"Song '{title}' is too short ({duration_ms/1000:.2f}s), skipping...")
                    continue

                # Select a random starting point (ensuring at least 30 seconds are available)
                max_start_point = duration_ms - 30000
                start_point = random.randint(0, max_start_point)

                # Extract 30-second segment
                segment = audio[start_point:start_point + 30000]

                # Split into first 15 seconds and last 15 seconds
                first_15s = segment[:15000]
                last_15s = segment[15000:]

                # Generate filenames
                filename_prefix = f"{processed_count:05d}_{clean_artist}_{clean_title}"
                first_15s_path = os.path.join(output_dir, f"{filename_prefix}_first15.mp3")
                last_15s_path = os.path.join(output_dir, f"{filename_prefix}_last15.mp3")

                # Export audio segments
                first_15s.export(first_15s_path, format="mp3")
                last_15s.export(last_15s_path, format="mp3")

                # Write metadata to CSV
                with open(csv_file, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([title, artist, first_15s_path, last_15s_path])

                # Add the (title, artist) tuple to the processed_songs set to prevent duplicates
                processed_songs.add((title, artist))

                processed_count += 1

                # Print progress
                if processed_count % 10 == 0:
                    print(f"Processed {processed_count}/{num_songs} songs (seen {songs_seen} songs, selection rate: {processed_count/songs_seen:.2%})")

            except Exception as e:
                print(f"Error processing song '{title}': {e}")
                continue

        except StopIteration:
            print("Reached the end of the dataset. Restarting from the beginning.")
            ds_iter = iter(ds)
        except Exception as e:
            print(f"Unexpected error: {e}")
            continue

    print(f"Finished processing {processed_count} songs (seen {songs_seen} songs, selection rate: {processed_count/songs_seen:.2%}).")

if __name__ == "__main__":
    main()
