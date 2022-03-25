# Communication in Multi-Objective Normal-Form Games
This repo consists of five different types of agents that we have used in our study of communication in multi-objective normal-form games. The settings that involve communication do this following a leader-follower model as seen in Stackelberg games. 
In our setup, agents switch in a round-robin fashion between being the leader and communicating something and being the follower and observing the communication.

## No communication setting
In this setting two agents play a normal-form game for a certain amount of episodes. This experiment serves as a baseline for all other experiments.

## Cooperative action communication setting
In this setting, agents communicate the next action that they will play. The follower uses this message to pre-update their policy. This setting is similar to best-response iteration and attempts to find the optimal joint policy.

## Self-interested action communication setting
This setting enables agents to act completely self-interested. This means that agents learn a distinct leading policy and best-response policy to every distinct message. For historical reasons, the self-interested action communication agent is called "competitive action agent" in this repository.

## Cooperative policy communication setting
This setting follows the same dynamics as the cooperative action communication setting, but communicates the entire policy instead of the next action that will be played.

## Optional communication setting
The last setting enables agents to learn when to communicate. All agents learn a top-level policy that learns when to communicate. They also have two low-level protocols: a no communication protocol that gets followed when opting out of communication and a communication protocol which is followed when choosing to communicate. Each previous protocol can be selected as this low level communication protocol. For historical reasons, this agent is called the "optional communication agent" in this repository.

## Getting Started

Experiments can be run from the `MONFG.py` file. There are 7 MONFGs available, having different equilibria properties under the SER optimisation criterion, using the specified non linear utility functions. You can also select the type of experiment to run and other parameters. 

## Cite
Please use the following citation if you use these algorithms in your work.
```
@misc{ropke2021preference,
      title={Preference Communication in Multi-Objective Normal-Form Games}, 
      author={Willem Röpke and Diederik M. Roijers and Ann Nowé and Roxana Rădulescu},
      year={2021},
      eprint={2111.09191},
      archivePrefix={arXiv},
      primaryClass={cs.GT}
}
```

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details


