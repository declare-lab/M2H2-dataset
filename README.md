# M2H2: A Multimodal Multiparty Hindi Dataset For Humor Recognition in Conversations

:zap: :zap: :zap: Baseline Codes will be released soon!

:fire::fire::fire: [Read the paper](https://arxiv.org/abs/2108.01260)

The M2H2 dataset is compiled from a famous TV show "Shrimaan Shrimati Phir Se" (Total of 4.46 hours in length) and annotated them manually. We make groups of these samples (utterances) based on their context into scenes. Each utterance in each scene consists of a label indicating humor of that utterance i.e., humor or non-humor. Besides, each utterance is also annotated with its speaker and listener information. In multiparty conversation, listener identification poses a great challenge. In our dataset, we define the listener as the party in the conversation to whom the speaker is replying. Each utterance in each scene is coupled with its context utterances, which are preceding turns by the speakers participating in the conversation. It also contains multi-party conversations that are more challenging to classify than dyadic variants.

# Data Format

![Alt text](dataset_samples.png?raw=true "Sample")

## Text Data

:fire::fire::fire: ***The ``Raw-Text/Ep-NUMBER.tsv`` acts as a master annotation file which does not only contain the textual data but also contains other metadata as described below. It also contains the manually annotated labels of the utterances. Using the Episode id and scene id, one can map the utterances in the ``Raw-Text`` folder to the corresponding audio and visual segments in ``Raw-Audio`` and ``Raw-Visual``. This should result in multimodal data. The ``Label`` column in the TSV files e.g., ``Raw-Text/Ep-NUMBER.tsv`` contains the desired manually annotated labels for each utterance.***

The text data are stored in TSV format. Each of the file is named as ``Raw-Text/Ep-NUMBER.tsv``. Here the ``NUMBER`` is episode number which one should use to map with the corresponding audio and visual segments. The text data contains the following fields:

```
Scenes: The scene id. It will match the corresponding audio and visual segments.
SI. No.: Utterance number.
Start_time: Start time of the utterance in the video.
End_time: End time of the utterance in the video.
Utterance: The spoken utterance.
Label: The annotated label of the utterance. This can either be humor or non-humor.
Speaker: The format is "Speaker,listener". It has the form of "Speaker_name,utterance_id" e.g., "Dilruba,u3" which means the speaker is Dilruba and he is responding to utterance no. 3. This is particularly useful to resolve coreferences in a multiparty conversation.
```
## Audio Data

Every episode has a dedicated folder e.g., ``Raw-Audio/22/`` contains all the annotated audio samples for Episode no. 22.

For every episode, each scene has a dedicated folder e.g., ``Raw-Audio/22/Scene_1`` contains all the annotated audio samples for Episode no. 22 Scene 1.

## Visual Data

Every episode has a dedicated folder e.g., ``Raw-Visual/22/`` contains all the annotated visual samples for Episode no. 22.

For every episode, each scene has a dedicated folder e.g., ``Raw-Visual/22/Scene_1`` contains all the annotated visual samples for Episode no. 22 Scene 1.

# Baselines

:zap: :zap: :zap: Baseline Codes will be released soon!

# Citation

Dushyant Singh Chauhan, Gopendra Vikram Singh, Navonil Majumder, Amir Zadeh,, Asif Ekbal, Pushpak Bhattacharyya, Louis-philippe Morency, and Soujanya Poria. 2021. [M2H2: A Multimodal MultipartyHindi Dataset For Humor Recognition in Conversations. In ICMI ’21: 23rd ACM International Conference on Multimodal Interaction](https://arxiv.org/abs/2108.01260), Montreal, Canada. ACM, New York, NY, USA, 5 pages.
