# ltprg

This repository contains code for learning to play reference games.  
The code relies on the [mungpy](https://github.com/forkunited/mungpy)
repository for generic utilities to munge the reference game data into a JSON 
format, vectorize this JSON into several tensor views of the data, and run 
training/evaluations of the learning models using this tensor data.   

The remainder of this README is organized into the following sections:

* [A summary of the contents of this repository](#summary-of-repository-content)
* [Details of how to setup the code environment](#setup-instructions)
* [Instructions for pre-processing reference game data](#how-to-preprocess-reference-game-data-for-modeling)
* [Instructions for training and evaluating models](#how-to-train-and-evaluate-models)

## Summary of repository contents

The repository is organized into a *config* directory containing configuration
files for modeling experiments, a *scripts* directory containing templates for 
shell scripts for running various tasks, an *examples* directory containing data 
examples, a *src/main* directory containing a library of classes and functions for 
training and evaluating models, and a *src/test* directory containing tests, 
experiments, and scripts that run the model training and evaluation functions.
This separation between the main libraries and the scripts/tests was inspired by
previous Java Maven projects, and might seem annoyingly over-complicated in leading
to unnecessarily deep directory structures that aren't very typical of 
Python projects.  It is true that this is annoying, but the structure has had the 
benefit of keeping the one-off scripty type things separated away from 
the main library code in *src/main*. 

### Data subdirectories

The data in *examples* is organized into the following directories:

* *games/csv/* - Source csv files from mturk containing reference game data for various games

* *games/json/* - Games converted to the JSON format described in the section below on
preprocessing the game data.  This is the form of the data used by the rest of the 
featurization/modeling code.

* *games/misc/* - Miscellaneous, relatively unimportant junk

* *games/splits/* - Files describing partitions of the data sets into train/dev/test 
partitions.

Note that the *colorGrids* sub-directories contain most of the relevant game data that
is used as of May 2018.  These contain the recently collected color grids from mturk, and also
merged with the colors data set from Monroe et al (2017).  In *games/json/colorGrids*,
*1* is the first batch of color grid data collected from mturk, *2* is a second batch,
and *color3* is the Monroe et al (2017) color data pushed into the same format as the
colorGrids.  Directories ending in *sua_speaker* contain the state-utterance-action 
format of the data described in the pre-processing sections of the README below.  The
full merged data set used by the models as of May 2018 is 
in *games/json/colorGrids/12_color3_sua_speaker* (but this mess of stuff will probably 
be re-organized after the data set is finalized).
The split for all this data is in *games/splits/colorGrids_12_color_merged_34_80_33_10_33_10*
(which represents a 34/33/33 train/dev/test split of the original colors data and an 
80/10/10 train/dev/test split of the color grid data)

### Python modules

The Python libraries in *src/main/py/ltprg* are organized into the following modules:

#### Experiment configuration parsing 

* *ltprg.config.rsa* - Functions for parsing configuration files (in *config*) into 
PyTorch RSA modules and evaluations (from *ltprg.model.rsa*)

* *ltprg.config.seq* - Functions for parsing configuration files (in *config*) into
PyTorch sequence model modules (from *ltprg.model.seq*)

#### Data manipulation and pre-processing

* *ltprg.data.curriculum* - Functions for re-ordering data sets according to 
training curricula (e.g. ordering of game rounds according to the lengths of speaker 
utterances)

* *ltprg.data.feature_helpers* - Helper functions for computing vectorized views
of the reference game JSON data.  These vectorized views are used as inputs to the 
learning models.

* *ltprg.data.feature* - Feature classes used by *ltprg.data.feature_helpers* to compute
vectorized views of the reference game JSON data.

#### Modules specific to particular reference games

* *ltprg.game.color.properties* - Some helper functions for manipulating data from the
color reference game (this was imported from another library, and was never fully 
integrated into this one, but a few of the functions here are used by the data 
preprocessing code). 

* *ltprg.game.color.eval* - Model evaluations specific to the color reference game 
(e.g. for outputting visualizations of learned meaning functions computed over 
*Hue x Saturation* color space)

* *ltprg.game.color.util* - Utilities specific to color games

#### Model components

* *ltprg.model.dist* - Represenations of probability distributions

* *ltprg.model.meaning* - Modules for computing RSA meaning functions

* *ltprg.model.obs* - Modules to compute over observations within RSA
when prior to conditioning the world and utterance priors (these are used within
*ltprg.model.rsa* as *observation_fns*, especially for SNLI RSA models and others
that condition the priors on embeddings from observed sequences)

* *ltprg.model.prior* - Modules for computing utterance and world priors within
RSA models

* *ltprg.model.rsa* - RSA modeling and evaluation modules

* *ltprg.model.seq_heuristic* - Heuristics to guide sequence model sampling and 
search procedures (e.g. used for sampling utterance prior supports that contain 
utterances which score highly according to literal listeners)

* *ltprg.model.seq* - Sequence model modules

#### Miscellaneous utilities

* *ltprg.util.file* - Utilities for managing file I/O

* *ltprg.util.img* - Utitilities for manipulating images

### Python scripts

Some scripts in *src/test/py* are:

#### Data manipulation and pre-processing

* *ltprg/data/make_game_csv_to_json.py* - Take games from mturk in csv 
format and convert them to JSON objects (see *examples/games/csv* and 
*examples/games/json* for examples)

* *ltprg/data/annotate_json_nlp.py* - Process utterances from
games (in JSON format) to produce annotations with NLP tools (e.g. tokenize, 
lemmatize, etc), and output new annotated games in JSON format

* *ltprg/data/make_sua.py* - Transform reference games in JSON format (with
one JSON object per game) into a data set with one state-utterance-action JSON 
object per round (representing single training examples for learning models)

* *ltprg/data/compute_utt_length_distribution.py* - Compute distribution
of utterance lengths across games

* *ltprg/data/make_wv_feature_lookup.py* - Builds and save a numpy matrix from 
word vectors stored in a feature set (this was useful for initializing sequence 
models with GLoVe embeddings for SNLI)

#### Scripts specific to color grid (and merged colors) reference games

The *ltprg/game/colorGrids* directory contains scripts particular to the 
color grids data set (and also the colors data set that has been converted
and merged into this same format).  There are older scripts specific to other 
reference games in other subdirectories of *ltprg/game*, but currently 
(as of May 2018), the only actively used scripts are the *colorGrids* ones.
They are as follows:

##### Data pre-processing

* *ltprg/game/colorGrids/data/annotate_with_targets.py* - Annotate the data 
with explicit representations of the target objects for each round that can
be easily featurized (this script is a bit of a hack).  Data that has been
run through this script contains *targetObj* fields containing representations
of the target objects.

* *ltprg/game/colorGrids/data/compute_stats.py* - Compute various statistics
over the data (e.g. human listener accuracy)

* *ltprg/game/colorGrids/data/convert_from_color.py* - Convert the Monroe et
al (2017) color data set into the same format as the color grid data.

* *ltprg/game/colorGrids/data/featurize_sua.py* - Compute vectorized 
representations of the data for use within learning models.

* *ltprg/game/colorGrids/data/make_syn_split.py* - Make train/dev/test 
partitions of the data

* *ltprg/game/colorGrids/data/merge_data.py* - Merge various data sets into a
single data set (e.g. merge the Monroe et al. (2017) color data with the
newer color grids data)

##### Modeling

* *ltprg/game/colorGrids/model/learn_RSA.py* - Train RSA models

* *ltprg/game/colorGrids/model/learn_SRSA.py* - Train RSA models that use
the same sequence model for both their meaning functions and utterance 
priors (this was an idea pursued briefly, and may be worthwhile to look
at again in the future)

* *ltprg/game/colorGrids/model/learn_S.py* - Train sequence models (i.e.
vanilla language models and S0 models)

* *ltprg/game/colorGrids/model/test_prior.py* - Compute examples of 
utterance priors under various settings to visualized and evaluate 
qualitatively

#### Results post-processing

* *ltprg/results/aggregate.py* - Take result files output by several 
training/evaluation jobs (e.g. under different seeds and hyper-parameter 
settings with a script like *ltprg/game/colorGrids/model/learn_RSA.py*), 
and merge them into a single tsv file.

#### Visualization and investigation of learned meaning functions

* *ltprg/meaning/* - This directory will eventually contain several 
scripts and tools for investigating learned meanings

#### Tests

* *ltprg/model/test_rsa.py* - Contains a test for the RSA PyTorch module

### Other stuff (e.g. game visualization)

The directory *src/html/viewData/* contains web pages used to visualize the 
data set.  Specifically, the color and colorGrids data, and also utterance 
priors output by *ltprg/game/colorGrids/model/test_prior.py* can be visualized 
using *src/html/viewData/colorGrids/view.html* (by just opening this file in a
browser, and using the form to select a JSON file representing a game or 
the utterance prior output)

## Setup instructions

To setup, first clone the repository:

    git clone https://github.com/forkunited/ltprg

You will also need to clone the mungpy repository that contains several data
munging and experiment utilities upon which ltprg relies:

    git clone https://github.com/forkunited/mungpy

Next, set your PYTHONPATH environment variable to include the paths to the
Python libraries within these cloned repositories:

    export PYTHONPATH=$PYTHONPATH:/path/to/ltprg/src/main/py/:/path/to/mungpy/src/main/py/

The code generally relies on PyTorch for training models, so install 
that if it is not already installed.  Currently, PyTorch version 0.3.0.post4 
is used.  You can download it at https://pytorch.org/previous-versions/.

## How to preprocess reference game data for modeling

Before running modeling experiments, it's useful to push all the reference
game data into a standard format that can be manipuated by models through a
common set of data structures.  This section describes how to push CSV
data generated by reference games into a more easily manipulable JSON format,
and then generate vectorized views of this data that are used by the learning
models.  The main steps in pipeline include:

1. [Converting CSV game data to the JSON format](#converting-game-data-to-the-json-format)
2. [Producing NLP annotations for the JSON data](#producing-nlp-annotations-for-json-game-data)
3. [Extracting state-utterance-action data sets from JSON data](#extracting-state-utterance-action-data-sets)
4. [Partitioning data into train-dev-test splits](#partitioning-data-for-training-and-evaluation)
5. [Computing and saving feature matrices from the data](#computing-and-saving-feature-matrices-from-the-data)
6. [Reloading saved feature matrices into memory](#reloading-saved-feature-matrices-into-memory)

Steps 1 though 5 can be performed just once for a new reference game data set,
and step 6 is used in experiments to load the processed data for modeling, etc. 
A template for a single script that performs steps 1 through 5 is given in 
[scripts/preprocess.sh](https://github.com/forkunited/ltprg/blob/master/scripts/preprocess.sh).  
To setup the pipeline, this script template should be copied and filled in with
paths to data that are specific to the local environment.  Each of the pre-processing
steps is described in some detail below.

(Note that the color and color grids data sets under *examples/games/json/colorGrids* have been
generated through additional steps to the pipeline described here.  These additional
steps were necessary for merging the color and color grids data into a single format. 
But in general, when new reference game data comes through a single CSV format, it's likely that
the steps given here should be at least nearly sufficient for pre-processing.)

### Converting game data to the JSON format

Pushing the data from all the reference games into a standard format 
allows for easy re-use of featurization and modeling code.  Currently, we use
the format shown in the JSON schema below.  There are 
more examples of this format from the color data set in
[examples/games/json/color](https://github.com/forkunited/ltprg/tree/master/examples/games/json/color).

```json
{
 "gameid" : "*unique game identifier string*",
 "records": [{ "roundNum": 1,
               "events": [
                   { "type": "*StateSubtype*", "time": 1476996301986, "..." : "..." },
                   { "type": "Utterance", "time": 1476996265147, "sender": "speaker",
                     "contents": "*Stuff said by speaker*"
                   },
                   { "type": "Utterance", "time": 1476996265180, "sender": "listener",
                     "contents": "*Stuff said by listener*"
                   },
                   { "type": "Utterance", "time": 1476996265190, "sender": "speaker",
                     "contents": "*More stuff said by speaker*"
                   },
                   { "..." : "..." },
                   { "type": "*ActionSubtype*", "time": 1476996267239, "..." : "..." }
               ]
             },
             { "roundNum": 2, "events": [ { "...": "..." } ]},
             { "roundNum": 3, "events": [ { "...": "..." } ]},
             { "..." : "..."}
           ]
}
```

Note that in the above schema, place-holder values are given between the
asterisks, and the "..." fields indicate that the object could contain more
fields.

In this format, each reference game is represented by a single JSON object
containing a *"gameid"* unique identifier field for the game, and a *"records"*
field that contains a list of numbered game-round objects.  Each round object
consists of a *"roundNum"* (round number) and a list of events.  Each event
is either a state, an utterance, or an action.  Each of these events can
have several game-specific dimensions with arbitrarily complicated substructure
determined by the dimensions of the events of a particular game.

There are two options for converting reference game data into this format.  The
first is to write a custom script to convert from your source format to the JSON
format.  Alternatively, if you have a CSV representation of your data, then you
can use
[test/py/ltprg/data/make_game_csv_to_json.py](https://github.com/forkunited/ltprg/blob/master/src/test/py/ltprg/data/make_game_csv_to_json.py)
to convert to the JSON format.  See the documentation at the top of that script
for details about the CSV format that this script expects.

### Producing NLP annotations for JSON game data

When the reference game data is in JSON format, the utterances from the game can
be pushed through the Stanford CoreNLP pipeline using the script at
[test/py/ltprg/data/annotate_json_nlp.py](https://github.com/forkunited/ltprg/blob/master/src/test/py/ltprg/data/annotate_json_nlp.py).
See the documentation at the top of that script for details.  Also, there are 
examples of NLP annotated JSON color data in
[examples/games/json/color_nlp](https://github.com/forkunited/ltprg/tree/master/examples/games/json/color_nlp)
(each line of each file contains a JSON object representing a game).

### Extracting state-utterance-action data sets

Several reference game learning models depend on training from examples
that consist of a single round represented by a game state, utterances, and 
an action.  For example, training
the RSA listener models to play the color reference game depends on having
one example per game round---with a state of three colors, 
the speaker utterances, and the target color referred to by the speaker.  The 
script at
[test/py/ltprg/data/make_sua.py](https://github.com/forkunited/ltprg/blob/master/src/test/py/ltprg/data/make_sua.py)
constructs state-utterance-action examples like this from the game data.  Also, there
are examples of state-utterance-action data for the color data set in
[examples/games/json/color_sua_speaker](https://github.com/forkunited/ltprg/tree/master/examples/games/json/color_sua_speaker).

### Partitioning data for training and evaluation

When training and evaluating models, it's useful to partition the data into 
train/dev/test sets---or other splits.  For this purpose, the
[mung.data.Partition](https://github.com/forkunited/mungpy/blob/master/src/main/py/mung/data.py)
class is useful for loading partitions of the data into memory, and splitting
data sets and feature matrices according to these partitions.  The class assumes
that the partitions are stored in the following JSON format:

```json
{
  "size": 948,
  "parts": {
    "train": { "1124-1": 1, "8235-6": 1, "..." : 1 },
    "dev": { "2641-2": 1, "4235-3": 1, "..." : 1 },
    "test": { "5913-4": 1, "1212-5": 1, "..." : 1 }
  }
}
```

The *"size"* field specifies the number of elements across all parts of the
partition, and the *"parts"* field contains the parts of the partition.  
Each part contains all
of the keys representing objects in the partition.  In the example shown above,
each key is a game ID.  The *mung.data.Partition*
can *split* a data set according to this partition and a "key function" that
maps datums from a data set to the keys in the partition.  Since the keys are
game IDs in the above example, the "key function" would need to map datums to
their game IDs to split a data set.

The script at 
[test/py/ltprg/data/make_split.py](https://github.com/forkunited/ltprg/blob/master/src/test/py/ltprg/data/make_split.py)
takes a path to a set of JSON game data, and uses mung.data.Partition to create a 
split based on game ID. There are many examples of data splits in
[examples/games/splits](https://github.com/forkunited/ltprg/tree/master/examples/games/splits).

### Computing and saving feature matrices from the data

Feature matrices can be computed from the state-utterance-action and game
data sets in the JSON format described above.  In particular, the
[mungpy.feature_helpers](https://github.com/forkunited/mungpy/blob/master/src/main/py/mung/feature_helpers.py)
module (from
[mungpy](https://github.com/forkunited/mungpy)) contains functions for
constructing feature matrices and sequences of feature
matrices (e.g. representing  utterances).   The features in the matrices can
represent values stored in the JSON data examples.  These values can be either
numerical values stored directly in properties of the JSON objects, or tokens from an
enumerable vocabulary (e.g. of utterance words/lemmas).  The following
functions from *mung.feature_helpers* are used to construct the matrices:

* *featurize_path_enum* : This constructs a feature matrix where each column  
represents a token from some enumerable vocabulary extracted from the data, and
each row represents the values of the features for a particular datum.  So, the
entry at row *i* and column *j* indicates whether token *j* occurs in datum *i*.  
As a more compact alternative to this sparse one-hot representation, this function
can also store the token index numbers directly in the matrix.  Prior to
constructing the matrix, this function also constructs the token vocabulary
by gathering all possible tokens from across the data set.

* *featurize_path_enum_seqs* : This works similar to *featurize_path_enum*,
except that it computes a sequence of matrices rather than a single matrix,
where matrix *i* contains a representation of token *i* in the sequence for each
datum.  If some datum sequences contain fewer elements than others, then they
are padded.  Currently, such matrix sequences are used to represent the sequences
of words in utterances across a data set.

* *featurize_path_scalars* : This constructs a feature matrix where each column
represents some numerical dimension from the examples in the data set.  

As an example of how these functions are used, consider the state-utterance-action
datums from the color data set.  Each of these datums has the following form:

```json
{
  "id": "6655-7_1_0", "gameid": "6655-7", "sua": 0, "roundNum": 1,
  "state": {
    "type": "StateColor",
    "lH_0": "248", "lH_1": "294", "lH_2": "101",
    "lS_0": "31", "lS_1": "48", "lS_2": "77",
    "lL_0": "50", "lL_1": "50", "lL_2": "50",
    "..." : "..."
  },
  "utterances": [
    {
      "type": "Utterance",
      "sender": "speaker", "contents": "bright blue", "time": 1476989293766,
      "nlp": {
        "type": "CoreNLPAnnotations",
        "lemmas": { "lemmas": [ "bright", "blue" ], "..." : "..." },
        "tokens": "...", "sents": "...", "pos": "..."
      }
    }
  ],
  "action": {
    "type": "ActionColor",
    "time": 1476989295346,
    "lClicked_0": "1", "lClicked_1": "0", "lClicked_2": "0"
  }
}
```

Note that in this example,  there are indicators of which color the listener
clicked in the *"lClicked_i"* fields of the *"action"* sub-object.  Given datums in
this form stored at *input_data_dir*, the following function will compute a
feature matrix containing rows of these *"lClicked_i"* indicators for each datum:

```python
mung.feature_helpers.featurize_path_scalars(
    input_data_dir, # Source data set
    join(output_feature_dir, "listener_clicked"), # Output directory
    partition_file, # Data partition
    lambda d : d.get("gameid"), # Function that partitions the data
    "listener_clicked", # Name of the feature
    ["action.lClicked_0", "action.lClicked_1", "action.lClicked_2"], # JSON paths to feature values
    init_data="train") 
```

The first two arguments to the function specify the input and output locations
on disk.  The third and fourth argument specify the location of a file storing
a data partition and the "key function" for partitioning the data (see the
section on [partitions](#data-partitons)), and the final *init_data*
specifies the part of this partition on which to *initialize* the features
(this is necessary when the feature initialization depends on some property of
the data, but should not depend on the values from the test data). 
The "listener_clicked" argument just gives a name for the feature.  Finally,
the list of "action.lClicked_0", "action.lClicked_1", and "action.lClicked_2"
specifies the JSON paths within the data from which to gather the feature
values.  This will construct a matrix where each row represents a
state-utterance-action example, and each the columns represent values from
"action.lClicked_0", "action.lClicked_1", and "action.lClicked_2".

As another example, the following function computes sequences of feature
matrices representing utterances from the state-utterance action data:

```python
mung.feature_helpers.featurize_path_enum_seqs(
    input_data_dir, # Source data set
    join(output_feature_dir, "utterance_lemma_idx"), # Output directory
    partition_file, # Data partition
    lambda d : d.get("gameid"), # Function that partitions the data
    "utterance_lemmas_idx", # Name of the feature
    ["utterances[*].nlp.lemmas.lemmas"], # JSON path into data examples
    15, # Maximum utterance length
    token_fn=lambda x : x.lower(), # Function applied to tokens to construct the vocabulary
    indices=True, # Indicates that indices will be computed instead of one-hot vectors
    init_data="train") 
```

The function first computes a vocabulary of lower-cased lemmas gathered from the
"utterances[\*].nlp.lemmas.lemmas" JSON path across the data set.  Then, it computes
a sequence of *15* padded feature matrices representing the lemmas of utterance 
tokens for each datum.

See
[test/py/ltprg/game/color/data/feature_sua.py](https://github.com/forkunited/ltprg/blob/master/src/test/py/ltprg/game/color/data/featurize_sua.py)
for some further examples of how feature matrices are constructed from the color
reference game data.

### Reloading saved feature matrices into memory

Assuming a state-utterance-action data set in *sua_data_dir* has been
featurized using the methods described in the previous section, the feature
matrices can be reloaded into memory from *features_dir* using code like the
following:

```python
D = MultiviewDataSet.load(
    sua_data_dir,
    dfmat_paths={
        "listener_clicked" : join(features_dir, "listener_clicked"),
        "listener_colors" : join(features_dir, "listener_colors"),
        "speaker_colors" : join(features_dir, "speaker_colors"),
        "speaker_observed" : join(features_dir, "speaker_observed"),
        "speaker_target_color" : join(features_dir, "speaker_target_color"),
        "speaker_target" : join(features_dir, "speaker_target")
    },
    dfmatseq_paths={
        "utterance_lemma_idx" : join(features_dir, "utterance_lemma_idx")
    })
partition = Partition.load(partition_file)
D_parts = D.partition(partition, lambda d : d.get("gameid"))
D_train = D_parts["train"]

batch = D_train.get_random_batch(5)
```

This loads feature matrices from each of the sub-directories under *features_dir*
as specified by the *dfmat_paths* argument, and feature matrix sequences from
the paths specified by the *dfmatseq_paths* argument.  The data is then split
according to the partition loaded from *partition_file* (see the
section on [partitions](#data-partitons)).  Sub-matrices representing a random
batch of 5 examples are extracted using the *get_random_batch* method on the
last line.  The returned *batch* object is a dictionary with keys for each
of the loaded feature matrices and feature matrix sequences.  So for example,
*batch["listener_clicked"]* will contain a matrix with 5 rows, and columns
representing where the listener clicked (as computed in the previous
  section).

See the unit tests at
[test/py/ltprg/game/color/data/validate_data.py](https://github.com/forkunited/ltprg/blob/master/src/test/py/ltprg/game/color/data/validate_data.py)
for more examples of how to access the data.  

## How to train and evaluate models

### FIXME
