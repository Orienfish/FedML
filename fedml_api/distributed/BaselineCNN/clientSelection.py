import random
import math
import numpy as np
import logging

class Tier(object):
    """Tier objects for client selection"""
    def __init__(self, client_id_list, probability, credits):
        self.client_id_list = client_id_list
        self.p = probability
        self.credits = credits
        self.mean_loss = 10

class ClientSelection(object):
    """Client selection decision making."""
    def __init__(self, n_clients, select_type, rounds, gamma):
        self.n_clients = n_clients
        self.select_type = select_type
        self.rounds = rounds
        self.gamma = gamma

        # Create loss and delay records
        self.est_delay = np.array([0.0 for _ in range(self.n_clients)])
        self.est_delay_sample_num = np.array([0 for _ in range(self.n_clients)])
        self.losses = np.array([0.0 for _ in range(self.n_clients)])
        self.grads = None
        self.num_samples = np.array([0 for _ in range(self.n_clients)])
        self.num_samples = np.reshape(self.num_samples, (self.n_clients, 1))
        self.grads_err_mat = np.zeros((self.n_clients, self.n_clients))
        self.dissimil_mat = np.zeros((self.n_clients, self.n_clients))

        # Perform light-weight profiling on clients
        if self.select_type == 'tier':
            self.tiers = self.tier_profiling()

    def update_loss_n_delay(self, losses, delays, client_id='all'):
        # if client_id = 'all', losses: numpy array of losses on all clients, (n,)
        #                       delays: numpy array of delays on all clients, (n,)
        # if client_id = int, losses: int of loss on client #client_id
        #                     num_samples: int of num_samples on client #client_id
        if client_id == 'all':  # First-time update
            self.losses = losses
            self.est_delay = (self.est_delay * self.est_delay_sample_num +
                              delays) / (self.est_delay_sample_num + 1)
            self.est_delay_sample_num += 1
        else:  # Update on client #client_id
            self.losses[client_id] = losses
            self.est_delay[client_id] = (self.est_delay[client_id] * self.est_delay_sample_num[client_id] +
                                         delays) / (self.est_delay_sample_num[client_id] + 1)
            self.est_delay_sample_num[client_id] += 1

    def update_grads(self, grads, num_samples, client_id):
        # if client_id = int, grads: numpy array of grads on client #client_id, (n,)
        #                     num_samples: int of num_samples on client #client_id
        if self.grads is None:  # First-time update
            grads_len = len(grads)
            self.grads = np.zeros((self.n_clients, grads_len))

        self.grads[client_id, :] = np.array(grads)
        self.num_samples[client_id] = num_samples
        self.avg_grad = np.sum(
            np.multiply(grads, self.num_samples), axis=0
        ) / np.sum(self.num_samples)  # (n,)

        self.grads_err_mat[client_id, :] = np.sum(
            np.square(self.grads - self.grads[client_id]), axis=1
        )
        self.grads_err_mat[:, client_id] = self.grads_err_mat[client_id, :]

        # Update disimilarity matrix
        self.dissimil_mat[client_id, :] = grads @ self.grads.T
        self.dissimil_mat[:, client_id] = self.dissimil_mat[client_id, :]
        np.fill_diagonal(self.dissimil_mat, 0.0)

        # print(client_id, grads[:10])
        # print(self.grads[:, :10])
        logging.info(self.dissimil_mat)

    def tier_profiling(self):
        # Sort clients by delay, fastest first
        sorted_clients_ids = sorted(np.arange(self.n_clients),
                                key=lambda i: self.est_delay[i])

        # Determine number of tiers
        est_clients_per_round = 5
        m = self.n_clients / est_clients_per_round

        if m < 5:
            m = math.floor(m)
        elif m <= 10:
            m = 5
        else:
            m = 10

        # Determine the credits / tier
        credits = math.ceil(self.rounds / m)

        # Give equal tier equal probability to begin
        p = 1/m

        # Place clients in each group
        clients_per_group = math.floor(self.n_clients/m)

        tiers = {}
        for i in range(0, m):
            if i != m-1:
                temp = sorted_clients_ids[clients_per_group * i : clients_per_group * (i+1)]
            else:
                temp = sorted_clients_ids[clients_per_group * i : ]
            tiers[i] = Tier(temp, p, credits)

        return tiers

    def tier_change_prob(self):
        selected_tier = self.last_select_tier
        mean = sum(
            [self.losses[i] for i in self.tiers[selected_tier].client_id_list])
        mean /= len(self.tiers[selected_tier].client_id_list)
        self.tiers[selected_tier].mean_loss = mean

        # tiers = [self.tiers[tier] for tier in self.tiers]

        #sort tiers highest loss first
        sorted_tiers = sorted(self.tiers, key=lambda t: self.tiers[t].mean_loss,
                              reverse=True)

        #count tiers with credits left
        credit_cnt = 0
        for tier in sorted_tiers:
            print("Tier Loss" + str(tier) + " : " + str(self.tiers[tier].mean_loss))
            print("Tier Credits" + str(tier) + " : " + str(self.tiers[tier].credits))

            if self.tiers[tier].credits > 0:
                credit_cnt = credit_cnt + 1

        #reset the probability for each tier
        D = credit_cnt * (credit_cnt - 1) / 2

        i = 0
        for tier in sorted_tiers:

            if self.tiers[tier].credits == 0:
                self.tiers[tier].p = 0
                continue
            elif D > 0:
                temp = (credit_cnt-i)/D
                if temp < 0:
                    temp = 0
                self.tiers[tier].p = temp
            else:
                temp = credit_cnt -i
                if temp < 0:
                    temp = 0
                self.tiers[tier].p = temp
            print("Tier " + str(tier) + " : " + str(self.tiers[tier].p))
            i = i + 1

    def select(self, select_num, flag_client_model_uploaded):
        available = np.array(flag_client_model_uploaded)
        selected = np.array([False for _ in range(self.n_clients)])

        if self.select_type == 'divfl':
            sample_client_ids = []
            while np.sum(selected) < select_num:
                # selected/not_selected: used in divFL, a numpy array of [True or False]
                # indicating whether each client is already selected
                not_selected = available & ~selected

                # cur_G: (not selected, 1), current error in approximating not selected clients
                if np.sum(selected) > 1:  # More than one selected client right now
                    cur_G = np.min(
                        self.grads_err_mat[not_selected][:, selected], axis=1,
                        keepdims=True)
                elif np.sum(selected) == 1:  # Only one selected client right now
                    cur_G = self.grads_err_mat[not_selected, selected]
                else:  # First selection, no client is selected right now
                    cur_G = np.max(self.grads_err_mat, axis=1, keepdims=True)
                # print(cur_G, cur_G.shape)

                # err_rdt: (not selected, not selected), reduction in error if selected one more client
                # err_rdt[i, j]: the reduction in error for approximating i if j is selected
                err_rdt = np.maximum(
                    cur_G - self.grads_err_mat[not_selected][:, not_selected],
                    0.0)
                # print(err_rdt, err_rdt.shape)

                # total_err_rdt: (not selected), total error reduction if select j from non-selected clients
                total_err_rdt = np.sum(err_rdt, axis=0)
                select_client_id = np.argmax(total_err_rdt)
                # print(total_err_rdt, select_client)

                # Update client availability status
                selected[select_client_id] = True
                sample_client_ids.append(select_client_id)

        else:  # first sort all candidates according to certain rules
            available_ids = np.arange(self.n_clients)[available]

            if self.select_type == 'random':
                # Select clients
                candidates_ids = available_ids
                random.shuffle(candidates_ids)

            elif self.select_type == 'high_loss_first':
                # Select the clients with largest loss and random latency
                candidates_ids = sorted(available_ids,
                                        key=lambda i: self.losses[i],
                                        reverse=True)

            elif self.select_type == 'short_latency_first':
                # Select the clients with short latencies and random loss
                candidates_ids = sorted(available_ids,
                                        key=lambda i: self.est_delay[i])

            elif self.select_type == 'short_latency_high_loss_first':
                # Get the non-negative losses and delays
                mean, var = np.mean(self.losses), np.std(self.losses)
                losses = (self.losses - mean) / var
                mean, var = np.mean(self.est_delay), np.std(self.est_delay)
                delays = (self.est_delay - mean) / var

                # Sort the clients by jointly consider latency and loss
                candidates_ids = sorted(available_ids,
                                    key=lambda i: losses[i] - self.gamma * delays[i],
                                    reverse=True)
                print([losses[i] for i in candidates_ids])
                print([self.gamma * delays[i] for i in candidates_ids])

            elif self.select_type == 'coreset':
                # Get the non-negative losses and delays
                if np.sum(available) > 1:
                    mean, var = np.mean(self.est_delay), np.std(self.est_delay)
                    delays = (self.est_delay - mean) / var

                    # Compute the gradient similarity and dissimilarity
                    eta = self.avg_grad @ self.grads.T  # (n,)
                    v = - np.sum(self.dissimil_mat, axis=1) / (self.n_clients - 1)  # (n,)
                    div = eta + v
                    mean, var = np.mean(div), np.std(div)
                    div = (div - mean) / var

                    # Sort the clients by jointly consider latency and loss
                    candidates_ids = sorted(available_ids,
                                        key=lambda i: div[i] - self.gamma * delays[i],
                                        reverse=True)
                    print([div[i] for i in candidates_ids])
                    print([self.gamma * delays[i] for i in candidates_ids])

            elif self.select_type == 'tier':
                # Select a tier based on probabilities
                tiers = [num for num in self.tiers]
                tier_prob = [self.tiers[num].p for num in self.tiers]
                selected_tier = random.choices(tiers, weights=tier_prob)[0]
                print('selected_tier: ', selected_tier)
                credits = self.tiers[selected_tier].credits
                while credits == 0:
                    selected_tier = random.choices(tiers, weights=tier_prob)[0]
                    credits = self.tiers[selected_tier].credits

                self.tiers[selected_tier].credits = credits - 1
                self.last_select_tier = selected_tier

                # Select candidates randomly from tier
                candidates_ids = self.tiers[selected_tier].client_id_list
                random.shuffle(candidates_ids)

            else:
                raise ValueError(
                    "client select type not implemented: {}".format(self.select_type))

            # Pick the first k clients
            sample_clients_ids = candidates_ids[:select_num]

        # print(sample_clients)
        return sample_clients_ids