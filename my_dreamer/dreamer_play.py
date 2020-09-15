import tensorflow as tf
import numpy as np
import time
import cv2

import pickle
import utils


class Play:
    def __init__(self, env, model_maker, config, training=True):
        self.env = env

        # self.act_space = self.env.action_space
        self.model = model_maker(self.env, training, config)
        self._c = config
        self.env.reset()
        self.ob, _, _, _ = self.env.step(
            [0]
        )  # whether it is discrete or not, 0 is proper
        # self.batch_size = 16
        self.batch_size = self._c.batch_size
        self.batch_length = (
            self._c.batch_length
        )  # when it is not model-based learning, consider it controling the replay buffer
        self.TD_size = 1  # no TD
        self.play_records = []
        self.act_repeat_time = self._c.action_repeat
        self.advantage = True
        self.total_step = 1
        self.exploration_rate = 0.0  # the exploration is in dreamer_net
        self.save_play_img = False
        self.RGB_array_list = []
        self.episode_reward = 0
        self.episode_step = 1  # to avoid devide by zero
        self.horizon = 15
        self.datadir = self._c.logdir / "episodes"
        # self._dataset = iter(self.load_dataset(self.datadir, self._c))
        self.prefill_and_make_dataset()

    def prefill_and_make_dataset(self):
        # since it casuse error when random choice zero object in self.load_dataset
        self.collect(using_random_policy=False, must_be_whole_episode=True)
        self._dataset = iter(utils.load_dataset(self.datadir, self._c))

    def act_repeat(self, env, act):
        collect_reward = 0
        for _ in range(self.act_repeat_time):
            ob, reward, done, info = self.env.step(act)
            collect_reward += reward

            if done:
                break

        return ob, collect_reward, done, info

    def TD_dict_to_TD_train_data(self, batch_data, advantage=False):
        # list of dict of batch_size,TD_size,{4}
        obs = []
        acts = []
        obp1s = []
        rewards = []
        dones = []
        discounts = []

        if advantage:

            for TD_data in batch_data:

                # TD_size,{4}
                TD_reward = 0
                ob = TD_data[0]["ob"]
                act = TD_data[0]["action"]
                obp1 = TD_data[-1]["obp1"]

                for i in range(self.TD_size):
                    TD_reward += TD_data[i]["reward"]
                done = TD_data[-1]["done"]
                discount = TD_data[-1]["discount"]

                # print("TD_reward:",TD_reward)

                obs.append(ob)
                acts.append(act)
                obp1s.append(obp1)
                rewards.append(TD_reward)
                dones.append(float(done))
                discounts.append(discount)

            return (
                np.array(
                    obs
                ),  # if self.backward_n_step > 1, [batch_size,backward_n_step+1,obaservation_size]
                np.array(acts),
                np.array(
                    obp1s
                ),  # if self.backward_n_step > 1, [batch_size,backward_n_step+1,obaservation_size]
                np.array(rewards),
                np.array(dones),
                np.array(discounts),
            )

        else:
            for TD_data in batch_data:
                # TD_size,{4}
                TD_reward = 0
                ob = TD_data[0]["ob"]

                time.sleep(0.01)
                for i in range(self.TD_size):
                    TD_reward += TD_data[i]["reward"]
                done = TD_data[-1]["done"]
                obs.append(ob)
                print("TD_reward:", TD_reward)
                rewards.append(TD_reward)
                dones.append(float(done))

            return np.array(obs), np.array(rewards), np.array(dones)

    def act_exploration(self, act):
        # TODO: epsilon greedy
        std = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3])
        noisy_act = np.random.normal(act, std)
        noisy_act = np.clip(noisy_act, -1, 1)

        return noisy_act

    def collect(self, using_random_policy=False, must_be_whole_episode=True):
        """
        collect end when the play_records full or episode end
        """
        trainsaction = {
            "ob": self.ob,
            "obp1": self.ob,
            "action": np.zeros(
                [
                    4,
                ]
            ),
            "reward": 0.0,
            "done": 0,
            "discount": np.array(1.0),
        }

        episode_record = [trainsaction]
        """
        I call this episode_record since when there's a done for playing, I must end collecting data for 
        capturing the end score of an episode, not to mix with next episode start when doing TD.
        It will start to train(in other words, break the loop) in two situation: 
        first, episode is done; second, the buffer is full(>self.batch_size * self.batch_length*self.TD_size).
        """
        # while len(episode_record) < self.batch_size * self.batch_length * self.TD_size:
        while True:  # stop only when episoe ends
            # episode = []
            # while True:
            if using_random_policy:
                act = self.model.random_policy_playing().numpy()  # to get batch dim

            else:

                processed_obs = utils.preprocess(
                    {"obs": np.array([self.ob]),"obp1s": np.array([self.ob]) ,"rewards": 0}, self._c
                )["obs"] # obp1s is for redundant here

                act = self.model.policy(processed_obs, training=True)[
                    0
                ].numpy()  # to get batch dim,

            if self._c.is_discrete:
                argmax_act = np.argmax(act, axis=-1)

            # # save play img

            # if self.total_step % 50000 == 0:
            #     self.save_play_img = True

            # if self.save_play_img == True:
            #     self.RGB_array_list.append(self.env.render())

            # if len(self.RGB_array_list) > 500:

            #     for i in range(len(self.RGB_array_list)):
            #         play_img = self.RGB_array_list[i]
            #         play_img = cv2.cvtColor(play_img, cv2.COLOR_RGB2BGR)

            #         cv2.imwrite("./train_gen_img/play_img_{}.jpg".format(i), play_img)
            #     self.RGB_array_list = []
            #     self.save_play_img = False

            ob, reward, done, info = self.act_repeat(self.env, argmax_act[0])
            self.total_step += 1

            trainsaction = {
                "ob": self.ob,
                "obp1": ob,
                "action": act[0],
                "reward": reward,
                "done": done,
                "discount": np.array(
                    1 - float(done)
                ),  # it means when done, discount = 1, else 0.
            }  # ob+1(obp1) for advantage method

            episode_record.append(trainsaction)
            # print(
            #     "collecting_data!!:",
            #     self.total_step,
            #     "len(episode_record):",
            #     len(episode_record),
            # )

            self.ob = ob

            # to coumpute average reward of each step
            self.episode_reward += reward
            self.episode_step += 1  # to avoid devide by zero

            # for first 100 batch, play just 500 step

            if self.total_step < 50000:
                if self.total_step % 500 == 0:
                    print("pre-set!")
                    done = True

            if done:
                print("game done!!")
                self.env.reset()
                self.ob, _, _, _ = self.env.step(
                    [0]
                )  # whether it is discrete or not, 0 is proper

                average_reward = self.episode_reward / self.episode_step

                # for dreamer, it need to reset state at end of every episode
                self.model.reset()

                # if len(episode_record) < self.TD_size:
                if (
                    len(episode_record)
                    < self.batch_size * self.batch_length * self.TD_size
                ):
                    print(
                        "it is shorter than a minimum episode length, abort this episode"
                    )
                    trainsaction = {
                        "ob": self.ob,
                        "obp1": self.ob,
                        "action": np.zeros(
                            [
                                4,
                            ]
                        ),
                        "reward": 0.0,
                        "done": 0,
                        "discount": np.array(1.0),
                    }  # ob+1(obp1) for advantage method
                    episode_record = [trainsaction]
                    if not must_be_whole_episode:
                        break
                else:
                    # done, break the loop for training
                    print("enough data!!")
                    break

        """
        move "TD_dict_to_TD_train_data" before update
        """
        if len(episode_record) > 1:
            # tranfer data to TD dict
            for i in range(
                (len(episode_record) // self.TD_size) + 1
            ):  # +1 for take the last step of play

                TD_data = episode_record[i * self.TD_size : (i + 1) * self.TD_size]
                if (
                    len(TD_data) != self.TD_size
                ):  # to deal with +1 causing not enough data of a TD size
                    print("reverse take for taking just to end of episode")
                    TD_data = episode_record[-self.TD_size :]
                self.play_records.append(TD_data)
                # so the structure of self.play_records is:
                # (batch_size* batch_length*TD_size or greater , TD_size)

            episode_record = []
            tuple_of_episode_columns = self.TD_dict_to_TD_train_data(
                self.play_records, True
            )

            dict_of_episode_record = {
                "obs": tuple_of_episode_columns[0],
                "actions": tuple_of_episode_columns[1],
                "obp1s": tuple_of_episode_columns[2],
                "rewards": tuple_of_episode_columns[3],
                "dones": tuple_of_episode_columns[4],
                "discounts": tuple_of_episode_columns[5],
            }

            # reset the inner buffer
            utils.save_episode(self.datadir, dict_of_episode_record)
            with self.model._writer.as_default():
                tf.summary.scalar(
                    "episode_reward",
                    tf.reduce_sum(tuple_of_episode_columns[3]),
                    step=self.model.total_step,
                )

            self.post_process_play_records()
        else:
            print("not enough data!!")

    def post_process_play_records(self):
        self.play_records = []

    def update(self):
        # this is single thread playing and processing
        # TODO: multi thread playing and processing
        data = next(self._dataset)

        # len(self.play_records) batch*batch_length*TD_size
        for i in range(
            (len(self.play_records) // self.batch_size) + 1
        ):  # +1 for take the last step of play
            print("updata_i:", i)
            batch_data = self.play_records[
                i * self.batch_size : (i + 1) * self.batch_size
            ]

            if (
                len(batch_data) != self.batch_size
            ):  # to deal with +1 causing not enough data of a batch size
                batch_data = self.play_records[-self.batch_size :]
                print("reversely take batch data")
            # print("batch_data:", len(batch_data))

            if self.advantage:
                # obs, obp1s, rewards, dones = self.TD_dict_to_TD_train_data(
                #     batch_data, advantage=self.advantage
                # )  # (batch_size, 57), (batch_size,)
                data = utils.preprocess(data, self._c)
                obs, obp1s, rewards, dones = (
                    data["obs"],
                    data["obp1s"],
                    data["rewards"],
                    data["dones"],
                )

                start_time = time.time()
                # rewards_mean = self.model.update_advantage(obs, obp1s, rewards, dones)
                rewards_mean = self.model.update_dreaming(obs, obp1s, rewards, dones)
                end_time = time.time()
                print("update time = ", end_time - start_time)

            else:
                obs, actions, rewards, dones, discounts = self.TD_dict_to_TD_train_data(
                    batch_data
                )  # (batch_size, 57), (batch_size,)
                # print("obs.shape:",obs.shape)
                # print("rewards:",rewards.shape)
                # rewards_mean = self.model.update(obs, rewards, dones)

                rewards_mean = self.model.update_dreaming(obs, actions, rewards, dones)

        # self.post_process_play_records()
        return rewards_mean

    def dreaming_update(self):
        """
        using collect function
        1. collect real play record(this should from collect function)
           get (batch,batch_length,feature_size) data

        when geting enough data
        2. using the real play record, batch by batch process
        3. in each batch process:
            world model stage
            a. do extract embed from data
            b. from embed to feat
            c. using feat to reconstruct observation
            d. get world model gradient

            actor stage
            e. do imagine step to horizon, for actor and critic:
            for example:
                from (batch,batch_length,feature_size)  to (batch, horizon,batch_length,feature_size)
            f. get value
            g. do dreamer lambda return
            h. maximize lambda return

            critic stage
            i. make value and lambda return closer as much as possible


        4. in each batch process, after imagine step, do update actor and critic
        """
        data = next(self._dataset)
        data = utils.preprocess(data, self._c)

        obs, actions, obp1s, rewards, dones, discounts = (
            data["obs"],
            data["actions"],
            data["obp1s"],
            data["rewards"],
            data["dones"],
            data["discounts"],
        )

        # print("obs:", obs.shape)  # (50, 10, 160, 160, 1)
        # print("actinons", actions.shape)
        # print("obp1s:", obs.shape)  # (50, 10, 160, 160, 1)
        # print("rewards:", rewards.shape)  # (50, 10)
        # print("dones:", rewards.shape)  # (50, 10)
        # print("discounts:", discounts.shape)

        start_time = time.time()
        rewards_mean = self.model.update_dreaming(
            obs, actions, obp1s, rewards, dones, discounts
        )
        end_time = time.time()
        # print("update time = ", end_time - start_time)

        self.post_process_play_records()
        return rewards_mean
