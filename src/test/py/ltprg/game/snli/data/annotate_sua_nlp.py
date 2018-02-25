import sys
import mung.nlp.corenlp

input_data_dir = sys.argv[1]
output_data_dir = sys.argv[2]

annotator = mung.nlp.corenlp.CoreNLPAnnotator('$.[state, utterance]', 'contents', 'nlp')
annotator.annotate_directory(input_data_dir, output_data_dir, id_key="id", batch=100)
