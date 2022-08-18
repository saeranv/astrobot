from google.cloud import translate
import os, sys
import numpy as np
from pprint import pprint as pp
ALPHA_REF = "abcdefghijklmnopqrstuvwxyz"

def translate_text(text_arr, project_id):
    """Translate text.

    Args:
        text_arr: 1D array of tuples, with each tuple representing
            one line of text via multiple characters:
            ```
            np.array(
                ['00:00:27,945 --> 00:00:32,783\n',
                 '[Naru, in Comanche] <i>Soobesükütsa tüa</i>\n',
                 '<i>pia mupitsl ikÜ kimai</i>.\n',
                 '\n',
                 '00:00:34,326 --> 00:00:39,748\n',
                 '[in English] <i>A long time ago, it is said,</i>\n',
                 '<i>a monster came here.</i>\n',
                 '\n'])
            ```
    """
    # Map True if alpha else False
    alpha_bool = np.array(
        [any(c.isalpha() for c in line)
         for i, line in enumerate(text_arr)],
        dtype=bool)

    client = translate.TranslationServiceClient()
    location = "global"
    parent = f"projects/{project_id}/locations/{location}"

    response = client.translate_text(
        request={
            "parent": parent,
            "contents": list(text_arr[alpha_bool]), # Pass just alphabetic lines
            "mime_type": "text/plain",
            "source_language_code": "en-US",
            "target_language_code": "ta",
        }
    )

    text_arr[alpha_bool] = \
        [translation.translated_text
         for translation in response.translations]

    return text_arr


if __name__ == "__main__":

    assert len(sys.argv) == 2, \
        (f"Invalid argument(s). Usage: python translate_srt.app PROJECT_ID, "
         f"got {sys.argv[1:]} in args.")
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
        _subs_en = np.array(subs_en[i0:i1], dtype=str)
        _subs_ta = translate_text(_subs_en, project_id)
        subs_ta.extend(_subs_ta)

    # Write file
    # for i in range(2):
    #     print(f"en: {subs_en[i]} ta: {subs_ta[i]}")

    with open(srt_fpath_ta, mode='w', encoding='utf-16-le') as f:
        f.writelines(subs_ta)

    print(f"Translated {len(subs_ta)} lines.")
    print(f"sed -n 100,110p data/in_tamil.srt >> cat 100-110 lines.")

