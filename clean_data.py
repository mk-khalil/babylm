import re
from pathlib import Path

# START_TOKEN = '<s>'
# END_TOKEN = '</s>'
# PADDING_TOKEN = '<pad>'

START_TOKEN = ''
END_TOKEN = ''
PADDING_TOKEN = ''

def _make_padding_sequence(seq_length):
    return ''.join([END_TOKEN] + seq_length * [PADDING_TOKEN])

def cleanup_simple_wikipedia(text, seq_length):
    pad_seq = _make_padding_sequence(seq_length)
    text = START_TOKEN + re.sub(r'\n\n', pad_seq + START_TOKEN, text) + pad_seq
    return text

def cleanup_wikipedia(text, seq_length):
    pad_seq = _make_padding_sequence(seq_length)
    text = re.sub(r'= = = (.+?) = = =\n', r'\1', text)
    lines = [line.strip() for line in text.splitlines()]
    text = START_TOKEN + re.sub(r'\n\n', pad_seq + START_TOKEN, '\n'.join(lines)[1:]) + pad_seq
    return text

def cleanup_qed(text, seq_length):
    # The handling of proper nouns and of parentheses isn’t perfect, but this is still an improvement over the base text
    punctuation_ex = re.compile(r'([.!?]\s*)')
    unimportant_chars_ex = re.compile(r'\(.*?\)|[.!?]')
    lines = []
    for line in text.splitlines():
        nchars = len(line)
        if nchars > 0:
            line_body = unimportant_chars_ex.sub('', line)
            f_upper = sum(c.isupper() for c in line_body) / len(line_body)
            if f_upper >= 0.5: # Mostly uppercase characters
                # Taken from https://stackoverflow.com/a/41662260
                split_on_punctuation = punctuation_ex.split(line.replace('l', 'I'))
                line = ''.join([sentence.capitalize() for sentence in split_on_punctuation])
        lines.append(line.strip())
    return START_TOKEN + '\n'.join(lines) + END_TOKEN + ''.join(seq_length * [PADDING_TOKEN])

def cleanup_extra_spaces(text):
    multiple_spaces_ex = re.compile(r'[ \t\u00A0]+')
    space_before_punctuation_ex = re.compile(r'[ \t\u00A0]([.,;!?])')
    text = multiple_spaces_ex.sub(' ', text)
    text = space_before_punctuation_ex.sub(r'\1', text)
    return text

def cleanup_bnc_spoken(text, seq_length):
    pad_seq = _make_padding_sequence(seq_length)
    text = cleanup_extra_spaces(text)
    text = START_TOKEN + re.sub(r'\n\n', pad_seq + START_TOKEN, text) + pad_seq
    return text

def cleanup_aochildes(text, seq_length):
    text = cleanup_extra_spaces(text)
    return START_TOKEN + text + _make_padding_sequence(seq_length)

def cleanup_cbt(text, seq_length):
    text = cleanup_extra_spaces(text)
    space_before_apostroph = re.compile(r"([\w\d])[ \t\u00A0](['’]\w)")
    #space_before_quote = re.compile(r"[ \t\u00A0](['’])")
    #space_after_quote = re.compile(r"([`])[ \t\u00A0]")
    #text = space_before_quote.sub(r'\1', text)
    #text = space_after_quote.sub(r'\1', text)
    text = space_before_apostroph.sub(r'\1\2', text)
    return START_TOKEN + text + _make_padding_sequence(seq_length)

def cleanup_children_stories(text, seq_length):
    # Sometimes one skipped line marks the beginning of a new story,
    # but sometimes it is present within a same story, which doesn’t
    # make it very useful for separating independent stories.
    return START_TOKEN + text + _make_padding_sequence(seq_length)

def cleanup_gutenberg(text, seq_length):
    # Overall, the text is clean, however some entries don’t seem
    # very useful, e.g. figure captions preceded by a number.
    # Not sure if we should remove them, because that would also
    # remove bullet lists which are otherwise consistent with the
    # surrounding text.
    # No start or end tokens because the text seems to be cut.
    return text + ''.join(seq_length * [PADDING_TOKEN])

def cleanup_open_subtitles(text, seq_length):
    # The text is mostly clean, apart from some subtitle credits
    # such as "Subtitles by ...".
    subtitle_credit_ex = re.compile(r'^.*subtitle.*$\n', re.MULTILINE | re.IGNORECASE)
    text = subtitle_credit_ex.sub('', text)
    return START_TOKEN + text + _make_padding_sequence(seq_length)

def cleanup_switchboard(text, seq_length):
    # No start or end tokens because the text seems to be cut.
    return text + ''.join(seq_length * [PADDING_TOKEN])


if __name__ == '__main__':

    DATA_ROOT = Path("./babylm_data")
    SEQ_LENGTH = 128 # this is a legacy parameter, it does not affect cleaning
    DATA_SPLITS = ['babylm_10M', 'babylm_dev']

    CLEANUP_FUNCTIONS = {
        'aochildes': cleanup_aochildes,
        'bnc_spoken': cleanup_bnc_spoken,
        'cbt': cleanup_cbt,
        'children_stories': cleanup_children_stories,
        'gutenberg': cleanup_gutenberg,
        'open_subtitles': cleanup_open_subtitles,
        'qed': cleanup_qed,
        'simple_wikipedia': cleanup_simple_wikipedia,
        'switchboard': cleanup_switchboard,
        'wikipedia': cleanup_wikipedia,
    }

    for split in DATA_SPLITS:
        
        INPUT_DIR = DATA_ROOT / split
        OUTPUT_DIR = DATA_ROOT / f'{split}_clean'

        OUTPUT_DIR.mkdir(exist_ok=True)

        train_files = [f for f in INPUT_DIR.iterdir() if f.is_file() and f.suffix in ['.train', '.dev']]

        for file in train_files:
            text = file.read_text()
            cleaned_text = CLEANUP_FUNCTIONS[file.stem](text, SEQ_LENGTH)
            (OUTPUT_DIR / file.name).write_text(cleaned_text)
            print(f"🧹 Cleaned '{file.name}' (size {len(text)} -> {len(cleaned_text)}) in {split}")