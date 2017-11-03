# alphago

This repo contains Keras code to recreate the neural network model used in DeepMind's AlphaGo Zero

See:

https://deepmind.com/blog/alphago-zero-learning-scratch/

http://nature.com/articles/doi:10.1038/nature24270

## TODO

It makes sense to reconstruct a minimally-working original code

What is missing right now:
* MCTS procedure
* Data processing procedures (encode the game into a tensor)
* The full proc that starts from a random state and incorporates a MCTS/update-NN to obtain better policies

*Feel free to make pull-requests :)*


## INSTALL

```
pip install -r requirements.txt
```