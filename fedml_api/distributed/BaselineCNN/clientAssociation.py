import numpy as np
import logging

class ClientAssociation(object):
    """Client association decision making."""
    def __init__(self, n_clients, n_gateways, asso_type, phi, model_name,
                 cls_num=None):
        # pref: [N, ], the preferred label of clients
        # cls_num: [N, ], the number of classes on clients
        self.n_clients = n_clients
        self.n_gateways = n_gateways
        self.asso_type = asso_type
        self.log_file = 'logca_{}'.format(model_name)
        self.phi = phi
        self.cls_num = cls_num

        self.est_delay = np.array([0.0 for _ in range(self.n_clients)])
        self.losses = np.array([0.0 for _ in range(self.n_clients)])
        self.R = np.array([0.0 for _ in range(self.n_clients)])
        self.grads = None
        self.num_samples = np.array([0 for _ in range(self.n_clients)])
        self.num_samples = np.reshape(self.num_samples, (self.n_clients, 1))
        self.dissimil_mat = np.zeros((self.n_clients, self.n_clients))

        self.prev_conn = None  # Record the last connection decision

    def update_loss_n_delay(self, loss, delay, client_id):
        # if client_id = int, losses: int of loss on client #client_id
        #                     num_samples: int of num_samples on client #client_id
        self.losses[client_id] = loss
        self.est_delay[client_id] = delay
        self.R[client_id] = 1 / delay

    def update_grads(self, grads, num_samples, client_id):
        # if client_id = int, grads: numpy array of grads on client #client_id, (n,)
        #                     num_samples: int of num_samples on client #client_id
        if self.grads is None:  # First-time update
            grads_len = len(grads)
            self.grads = np.zeros((self.n_clients, grads_len))

        self.grads[client_id, :] = np.array(grads)
        self.num_samples[client_id] = num_samples
        self.avg_grad = np.sum(
            np.multiply(self.grads, self.num_samples), axis=0
        ) / np.sum(self.num_samples)  # (n,)

        # Update disimilarity matrix
        self.dissimil_mat[client_id, :] = grads @ self.grads.T
        self.dissimil_mat[:, client_id] = self.dissimil_mat[client_id, :]
        np.fill_diagonal(self.dissimil_mat, 0.0)

        # print(client_id, grads[:10])
        # print(self.grads[:, :10])
        # logging.info(self.dissimil_mat)

        self.eta = (self.avg_grad @ self.grads.T).reshape((-1))  # (N, )
        self.v = - np.sum(self.dissimil_mat, axis=1) / (self.n_clients - 1)  # (N,)

        self.u = self.eta + self.v

    def solve(self):
        """
        Call the gurobi solver to solve the integer linear program of
        client association.

        Returns:
            conn: [N, G] matrix, decided connection
        """
        all_ids = np.arange(self.n_gateways)
        if self.asso_type == 'random':
            conn = []
            for i in range(self.n_clients):
                if self.cls_num is not None:
                    # Biased device-gateway matching depending on class num
                    match_book = {
                        1: 0,
                        2: 1,
                        3: 1,
                        4: 2,
                        5: 2
                    }
                    gateway_id = match_book[self.cls_num[i]]
                else:
                    gateway_id = np.random.choice(all_ids)

                conn.append(np.eye(self.n_gateways)[gateway_id])

            conn = np.array(conn, dtype=np.int)

        elif self.asso_type == 'gurobi':
            import gurobipy as gp
            from gurobipy import GRB

            # Create model and variables
            model = gp.Model('clientAssociation')
            model.setParam('MIPGap', 0.01)
            model.setParam('Timelimit', 100)
            vars = model.addMVar(shape=(self.n_clients, self.n_gateways), vtype=GRB.BINARY, name='vars')
            slack_simil = model.addVar(vtype=GRB.CONTINUOUS, name='slack_simil_var')
            slack_thpt = model.addVar(vtype=GRB.CONTINUOUS, name='slack_thpt_var')

            # Set objective
            model.setObjective(slack_simil - self.phi * slack_thpt,
                               GRB.MAXIMIZE)

            # Each value appears once per row
            model.addConstrs((sum(vars[i, :]) == 1
                              for i in range(self.n_clients)), name='unique')

            # Each value is restricted by the feasible connections
            #model.addConstrs((vars[:, j] <= conn_ub[:, j] for j in range(G)),
            #                 name='inclusive')

            # Add constraint for slack similarity
            model.addConstrs((self.u @ vars[:, j] >= slack_simil
                              for j in range(self.n_gateways)),
                              name='slack_simil')

            # Add constraint for slack delay
            model.addConstrs((self.R @ vars[:, j] <= slack_thpt
                              for j in range(self.n_gateways)),
                             name='slack_thpt')

            # Optimize model
            model.optimize()

            conn = np.array(vars.X, dtype=np.int)
            # print(conn)
            logging.info('Obj: {}'.format(model.ObjVal))


        else:
            raise ValueError(
                "client association type not implemented: {}".format(self.asso_type))

        logging.info('Obj 1: {}'.format(self.u @ conn))
        logging.info('Obj 2: {}'.format(self.R @ conn))

        with open(self.log_file, 'a') as f:
            f.write('Obj 1: {}\n'.format(self.u @ conn))
            f.write('Obj 2: {}\n'.format(self.R @ conn))
            for j in range(self.n_gateways):
                f.write('Gw {}: {} clients\n'.format(
                    j, np.sum(conn[:, j])
                ))
            f.write('\n')

        self.prev_conn = conn  # Update the previous connections
        print(conn)

        return conn