'''Tree utilities taken from https://gist.github.com/willwhitney/dd89cac6a5b771ccff18b06b33372c75'''
import numpy as np
import jax
from jax import numpy as jnp


def tree_stack(trees):
    '''Stacks a list of trees.'''
    return jax.tree.map(lambda *v: jnp.stack(v), *trees)

def tree_unstack(tree):
    '''Unstacks a tree into a list of trees.'''
    leaves, treedef = jax.tree.flatten(tree)
    return [treedef.unflatten(leaf) for leaf in zip(*leaves, strict=True)]


if __name__ == "__main__":
    '''Testing the tree_stack and tree_unstack functions.'''
    def make_tree():
        sizes = ((1, 2), (3, 1), (3,))
        make_leaf_np = lambda i: np.random.uniform(size=sizes[i])
        make_leaf = lambda i: jnp.array(make_leaf_np(i))
        return ((make_leaf(0), make_leaf(1)), make_leaf(2))
    trees = [make_tree() for _ in range(3)]
    print("Before")
    print(trees)

    print("\nStacked")
    stacked = tree_stack(trees)
    print(stacked)

    print("\nUnstacked")
    unstacked = tree_unstack(stacked)
    print(unstacked)