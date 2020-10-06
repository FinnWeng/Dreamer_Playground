# Struture Detail

![alt text](https://github.com/FinnWeng/Dreamer_Playground/blob/master/common/structure_forward.PNG "4to1")

Above how an action comes out. It relates to cell state which record GRU state, action of previous round, and embed from current observation.




![alt text](https://github.com/FinnWeng/Dreamer_Playground/blob/master/common/structure_world_model.PNG "4to1")

The flow of world model is most sophiscated. Upper parts is just like action production. However, whether to include observation makes two kinds of comeout: prior and posterior. Posterior has observation involved, while prior do not.
The posterior is then been used to produce two losses: KL divergence between prior and posterior, and loss of image and reward between prediction and real one from environment. 




![alt text](https://github.com/FinnWeng/Dreamer_Playground/blob/master/common/structure_action_function.PNG "4to1")

This shows how Dreamer maximize its return in order to get higher score.




![alt text](https://github.com/FinnWeng/Dreamer_Playground/blob/master/common/structure_value_function.PNG "4to1")

This show how Dreamer try to predict lambda return of action function, not the real reward.