from google.cloud import translate
import os, sys
import numpy as np


def translate_text(text_arr, project_id):

    client = translate.TranslationServiceClient()
    location = "global"
    parent = f"projects/{project_id}/locations/{location}"

    response = client.translate_text(
        request={
            "parent": parent,
            "contents": text_arr,
            "mime_type": "text/plain",
            "source_language_code": "en-US",
            "target_language_code": "ta",
        }
    )

    return [translation.translated_text
            for translation in response.translations]


if __name__ == "__main__":

    print(sys.argv)
    assert len(sys.argv) == 2, \
        "Invalid argument(s). Usage: python translate_srt.app PROJECT_ID"
    project_id = sys.argv[1]
    print(f"Translating with PROJECT_ID: {project_id}")

    # Read the file
    srt_fpath_en = os.path.join(os.getcwd(), "data", "in_english.srt")
    srt_fpath_ta = os.path.join(os.getcwd(), "data", "in_tamil.srt")
    assert os.path.exists(srt_fpath_en)

    # Open file
    with open(srt_fpath_en, mode='r') as f:
        subs_en = f.readlines()

    N = len(subs_en)

    # Translate
    subs_ta = []
    for i in range(0, N, 1024):
        i0, i1 = i, i+1024
        i1 = N if i1 >= N else i1
        subs_ta.extend(translate_text(subs_en[i0:i1], project_id))
        if i > 0: break

    # Write file
    for i in range(2):
        print(f"en: {subs_en[i]} ta: {subs_ta[i]}")

    with open(srt_fpath_ta, mode='w', encoding='utf-16-le') as f:
        f.writelines(subs_ta)

    print(f"Translated {len(subs_ta)} lines.")

