# ltprg

This repository contains code for learning to play reference games.  The
repository is organized into an *examples* directory containing data examples,
a *src/main* directory containing classes and functions for training and
evaluating models, and a *src/test* directory containing tests, experiments,
and scripts that run the model training and evaluation functions.

The library relies on the [mungpy](https://github.com/forkunited/mungpy)
repository at for munging the reference game data into a common format that
can be featurized for use by the learning models.  Some
instructions for how to push reference game data through this pipeline are
given below.  The main steps in this pipeline described below include:

1. [Converting game data to the JSON format](#converting-game-data-to-the-json-format)
2. [Extracting a state-utterance-action data set](#extracting-a-state-utterance-action-data-set)
3. [Computing and saving feature matrices from the data](#computing-and-saving-feature-matrices-from-the-data)
4. [Reloading saved feature matrices into memory](#reloading-saved-feature-matrices-into-memory)

## Converting game data to the JSON format

Pushing the data from all the reference games into a standard format will
allow for easy re-use of featurization and modeling code.  Currently, we use
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
format.  The second option is to put your data into an alternative CSV
representation, and then use
[test/py/ltprg/data/make_game_csv_to_json.py](https://github.com/forkunited/ltprg/blob/master/src/test/py/ltprg/data/make_game_csv_to_json.py)
to convert to the JSON format.  See the documentation at the top of that script
for details about the CSV format.

### Producing NLP annotations for JSON game data

When the reference game data is in JSON format, the utterances from the game can
be pushed through the Stanford CoreNLP pipeline using the script at
[test/py/ltprg/data/annotate_json_nlp.py](https://github.com/forkunited/ltprg/blob/master/src/test/py/ltprg/data/annotate_json_nlp.py).
See the documentation at the top of that script for details.

### Data partitions

When training and evaluate models, it's useful to partition the data into train/test
sets, or other splits.  For this purpose, the
[mung.data.Partition](https://github.com/forkunited/mungpy/blob/master/src/main/py/mung/data.py#L227)
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
partition, and the *"parts"* field contains the parts.  Each part contains all
of the keys representing objects in the partition.  In the example shown above,
each key is a game ID.  The *mung.data.Partition*
can *split* a data set according to this partition and a "key function" that
maps datums from a data set to the keys in the partition.  Since the keys are
game IDs in the above example, the "key function" would need to map datums to
their game IDs to split a data set.

## Extracting a state-utterance-action data set

Several reference game learning models depend on training from examples
that consist of a game state, utterances, and an action.  For example, training
the RSA listener models to play the color reference game depends on having
one example per game round, consisting of a state of three colors, 
the speaker utterances, and the color that the listener clicked after
hearing the utterances.  The script at
[test/py/ltprg/data/make_sua.py](https://github.com/forkunited/ltprg/blob/master/src/test/py/ltprg/data/make_sua.py)
constructs state-utterance-action examples like this from the game data.

## Computing and saving feature matrices from the data

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

## Reloading saved feature matrices into memory

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
for more examples of how to access the data.  The code for the
[S0](https://github.com/forkunited/ltprg/blob/master/src/test/py/ltprg/model/s0.py)
model also shows some more examples of how to use the methods provided by the
library.

## Notes for Running on Stanford Sherlock Servers
1) Load python module: `module load python/2.7.13`
2) Install necessary packages: `pip2.7 install --user requirements.txt`
3) If some requirement fails to import when running the code, update the dependency appropriately.
 Sometimes you have to uninstall and reinstall the package for it to import correctly.
4) To monitor batch runs with visdom run `ssh -N -f -L localhost:8097:localhost:8097 <sudnetid>@login.sherlock.stanford.edu` on your local machine to enable port forwarding.

