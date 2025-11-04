import os

from youtube_transcript_api import YouTubeTranscriptApi

video_id="Gfr50f6ZBvo"
ytt_api = YouTubeTranscriptApi()
fetched_transcript = ytt_api.fetch(video_id,languages=['en'])

transcript = " ".join(chunk["text"] for chunk in fetched_transcript.to_raw_data())


print(transcript)

# saving the transcript to a text file 

current_dir=os.getcwd()

documents_save_path=os.path.join(current_dir,"documents")

if not os.path.exists(documents_save_path):
    os.makedirs(documents_save_path)
with open(os.path.join(documents_save_path,"transcript.txt"),"w",encoding="utf-8") as f:
    f.write(transcript)
print(f"Transcript saved to {os.path.join(documents_save_path,'transcript.txt')}")
