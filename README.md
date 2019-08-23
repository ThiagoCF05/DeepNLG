# Pipeline vs. End-to-End Architecture to Neural Data-to-Text

This is the code used to obtained the results reported in the manuscript "Neural data-to-text generation: 
A comparison between pipeline and end-to-end architectures"

The file `main.sh` is the main script of this project. You may run it to extract the intermediate representations
from the data, to train the models, to evaluate each step of the pipeline approach (reported in Section 6 of the paper) 
as well as to generate text from the non-linguistic approach based on each model (reported in Section 7 of the paper).

To run the script, first install the Python dependencies by running the following command:

`
pip install > requirements.txt
`

Then update the root and the dependecies path on [vars](scripts/vars). This code has as dependencies 
[Moses](https://github.com/moses-smt/mosesdecoder), [Nematus](https://github.com/EdinburghNLP/nematus) and
[Subword NMT](https://github.com/rsennrich/subword-nmt). Once the paths are set, the script can be executed:

`
./main.sh
`

The augmented version of the WebNLG corpus is available [here](versions/v1.4). To see information about the evaluation, 
go to [the evaluation folder](evaluation/).