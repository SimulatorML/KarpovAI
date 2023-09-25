import os
import json
from typing import List
from pytube import YouTube
from parsers.channel_parser import ChannelParser


def get_video_info(video_url: str, path_to_save: str) -> dict:
    """
    Get info from Youtube-video - title, description, audio track
    Parameters
    ----------
    url: str
      url of Youtube-video
    path_to_save: str
      path for saving audio track

    Returns
    -------
    dict
      keys:
      - "title": str
      - "description": str
      - "audio_path": str
        saved audio path
    """
    yt = YouTube(video_url)
    audio_name = yt.video_id + ".mp4"
    audio_path = os.path.join(path_to_save, audio_name)
    title = yt.title
    print(f"Downloading {title}...")
    # The next string 'yt.streams...' must be before the following.
    # Surprisingly, but the fact is, if this is not observed,
    # then it is impossible to get a description.
    # Download mp4 audio file:
    yt.streams.filter(only_audio=True).first().download(path_to_save, audio_name)
    description = yt.description
    return {"url": video_url, "title": title, "description": description, "audio_path": audio_path}


def get_video_urls(channel_url: str) -> List[str]:
    """Get list of all video urls from youtube-channel"""
    channel = ChannelParser(channel_url)
    return channel.video_urls


def download_channel_audio_track(
    url_of_video: str, path_to_save_audio: str, json_video_info_path: str = None
) -> str:
    """
    Download audio track from YouTube-video and save info to json.
    Return path to downloaded audio track
    """
    if os.path.exists(json_video_info_path):
        with open(json_video_info_path, "r", encoding="utf-8") as f:
            video_info = json.load(f)
    else:
        video_info = []
    url_info = get_video_info(url_of_video, path_to_save_audio)
    print(f"Path to mp4 file: {url_info['audio_path']}\n")

    video_info_item = {"url": [], "title": [], "description": [], "audio_path": [], "text": []}
    for key, value in url_info.items():
        video_info_item[key].append(value)
    video_info.append(video_info_item)

    with open(json_video_info_path, "w", encoding="utf-8") as f:
        json.dump(video_info, f, ensure_ascii=False, indent=4)

    return video_info_item["audio_path"][0]


# def download_channel_audio_tracks(
#     channel_url: str,
#     path_to_save_audio: str,
#     json_video_info_path: str,
#     url_file_path: str = None,
# ) -> Dict[str, List[str]]:
#     """
#     This function allows you to download both all audio tracks of a video
#     from a YouTube channel, and download new ones using a list of previously downloaded audio.
#
#     Returns dict with info about new downloaded audio tracks.
#     It is necessary for further processing of downloaded audio.
#
#     Note: now the link to the channel has the form https://www.youtube.com /@{name_channel}.
#     In the function, it is necessary to submit a link of the form
#     https://www.youtube.com/c /{name_channel}
#     Parameters
#     ----------
#     channel_url: str
#       url of YouTube channel
#     path_to_save_audio: str
#       folder for downloaded audio
#     json_video_info_path: str
#       The path to the json file with info
#       about the saved audio tracks of a videos
#     url_file_path: str
#       The path to the txt file with
#       previously downloaded audio
#
#     Returns
#     -------
#     Dict[str, List[str]]
#       Dict with info about new downloaded audio tracks
#
#     """
#     # Read an existing dictionary of video info, or create a new one
#     if os.path.exists(json_video_info_path):
#         with open(json_video_info_path, "r", encoding="utf-8") as f:
#             video_info = json.load(f)
#     else:
#         video_info = {"title": [], "description": [], "audio_path": []}
#
#     # Dict for new crawling videos
#     crawling_videos = {"title": [], "description": [], "audio_path": []}
#
#     # Determine the set of downloaded videos
#     downloaded_videos = set()
#     if url_file_path is not None and os.path.exists(url_file_path):
#         with open(url_file_path, "r", encoding="utf-8") as f:
#             for line in f:
#                 downloaded_videos.add(line.strip())
#
#     # Determine the set of videos to download
#     new_videos = set(get_video_urls(channel_url)) - downloaded_videos
#
#     # If necessary, we add new videos to the file
#     if url_file_path is not None:
#         with open(url_file_path, "a", encoding="utf-8") as f:
#             for video in new_videos:
#                 f.write(video + "\n")
#
#     # Download new videos and add information about them to the dictionary
#     print(f"Need to download {len(new_videos)} videos\n")
#     for idx, url in enumerate(new_videos, 1):
#         print(f"{idx} video.")
#         url_info = get_video_info(url, path_to_save_audio)
#         print(f"Path to mp3 file: {url_info['audio_path']}\n")
#         for key, value in url_info.items():
#             video_info[key].append(value)
#             crawling_videos[key].append(value)
#         break
#
#     # Overwriting the json file with video info or writing a new one
#     with open(json_video_info_path, "w", encoding="utf-8") as f:
#         json.dump(video_info, f)
#
#     return crawling_videos


if __name__ == "__main__":
    _ = download_channel_audio_track(
        "https://www.youtube.com/watch?v=OXtOhjeiTzw", "../audio", "../video_info.json"
    )

    _ = download_channel_audio_track(
        "https://www.youtube.com/watch?v=9W1v-DkXriY", "../audio", "../video_info.json"
    )
#  _ = download_channel_audio_tracks(
#      "https://www.youtube.com/c/karpovcourses",
#      "audio",
#      "video_info.json",
#      "urls_of_channel_videos.txt",
#  )
