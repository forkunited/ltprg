import sys
import time
import mung.nlp.corenlp

"""
Annotates utterances from reference games in JSON format with NLP annotations
from the CoreNLP pipeline.  The script will extract the "contents" fields
of all "Utterance" events in the games, and run the text through the CoreNLP
pipeline to produce tokenizations, sentence segmentations, lemmatizations, and
PoS tags.  The resulting annotations will be stored in an "nlp" field of the
"Utterance" events within the resulting JSON objects in the output directory.

See "examples/games/json/color" for example input data, and
"examples/games/json/color_nlp" for example annotated output data.

Args:
    input_dir (:obj:`str`): Input directory containing JSON game data
    output_dir (:obj:`str`): Output directory in which to store JSON game data
        with NLP annotations
"""

input_data_dir = sys.argv[1]
output_data_dir = sys.argv[2]

annotator = mung.nlp.corenlp.CoreNLPAnnotator('$.records[*].events[?type = "Utterance"]', 'contents', 'nlp')
annotator.annotate_directory(input_data_dir, output_data_dir, id_key="gameid")
