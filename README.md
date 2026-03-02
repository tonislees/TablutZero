# ChessZero
A fully vectorized AlphaZero implementation for Chess using JAX, Flax, mctx, and Pgx.

## Instructions for training
1. Create a virtual environment for python and run:
```
pip install requirements.txt
```
2. Install BayesElo program by running these commands in the project root:
```
curl -O https://www.remi-coulom.fr/Bayesian-Elo/bayeselo.tar.bz2
tar -xjf bayeselo.tar.bz2
cd BayesElo
make
mv bayeselo ../
cd ..
rm -rf bayeselo.tar.bz2 BayesElo
```
3. Create a file called `config.yaml` and copy `config.template.yaml` contents into it.
4. Run `train.py`