# Dreamer_Playground

Here are my implementation of Dreamer, a Model-based Reinforment Learning algorithm from DREAM TO CONTROL: LEARNING BEHAVIOR BY LATENT IMAGINATION.

In this repo, I play my own implementation, official implementation and also some modification of official one.

# My Implementation


Here's the result of Dreamer for SpaceInvaders:

This is just result after 6 hours training, which means it converges super fast for image input!
![alt text](https://github.com/FinnWeng/Dreamer_Playground/blob/master/common/SpaceInvaders_play.gif "Dreamer_SpaceInvaders")

And the episode reward of it shows obvious ascending.

![alt text](https://github.com/FinnWeng/Dreamer_Playground/blob/master/common/SpaceInvaders_episode_rewards.PNG "4to1")



Here is result of my own implementation for breakout:

![alt text](https://github.com/FinnWeng/Dreamer_Playground/blob/master/common/4to1.png "4to1")

These four charts show how three loss of Dreamer and reward of environment changed. From left to right, up to down, there are Action Function Loss, Value Function Loss, World-Model Loss and episode reward.

![alt text](https://github.com/FinnWeng/Dreamer_Playground/blob/master/common/play_image_one_round.png "play_image_one_round")

After 100,000 times update, I confirm that a situation mentioned in paper happened: for discrete action space, Dreamer will get inferior result, lower than the result of model-free methods.

In the view of reward, it shows that Actor got mean reward of 3.5 almost every time, but barely got 0 score. This gives me a thought that Actor finds a local optimum in no time. And this optimum indeed make Actor got 3.5 point every time. 

Dreamer concludes a policy really fast, but it is not a good one. Seeing the play image, we could find Actor always choose to move toward left side of screen.

# Official Implementation

Besides implementation by my own, I also try origin implementation.

Here is the result of original dreamer:

![alt text](https://github.com/FinnWeng/Dreamer_Playground/blob/master/common/dreamer_origin.PNG "A3C with dreamer")


And here is the result of A3C modification of original dreamer:

![alt text](https://github.com/FinnWeng/Dreamer_Playground/blob/master/common/A3C_dreaming.PNG "A3C with dreamer")


