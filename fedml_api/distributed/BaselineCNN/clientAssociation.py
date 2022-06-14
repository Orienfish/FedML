import numpy as np
import logging

class ClientAssociation(object):
    """Client association decision making."""
    def __init__(self, asso_type, model_name,
                 pref=None, cls_num=None, labels=None):
        # pref: [N, ], the preferred label of clients
        # cls_num: [N, ], the number of classes on clients
        self.asso_type = asso_type
        self.log_file = 'logca_{}'.format(model_name)
        self.pref = pref
        self.labels = labels
        self.cls_num = cls_num
        self.label_num = len(labels)

    def solve(self, conn_ub, grads=None, num_samples=None, R=None,
              R_ub=None, phi=None):
        """
        Call the gurobi solver to solve the integer linear program of
        client association.
        Suppose there are N devices and G gateways.

        Args:
            conn_ub: [N, G] matrix, feasible connections
            grads: [N, # of weights], latest gradients at N clients
            num_samples: [N], number of samples at N clients
            R: [N, G] matrix, throughput of all possible links between
                device and gateway
            R_ub: [G], upper bound of throughput at each gateway
            phi: a scalar weight for the throughput term in the obj

        Returns:
            conn: [N, G] matrix, decided connection
        """
        N = conn_ub.shape[0]
        G = conn_ub.shape[1]

        if grads is not None:
            # Compute learning utility
            num_samples = num_samples.reshape((-1, 1))  # (N, 1)
            global_grad = np.sum(
                np.multiply(grads, num_samples), axis=0
            ) / np.sum(num_samples)  # (N,)
            eta = (global_grad @ grads.T).reshape((-1))  # (N, )

            # Update disimilarity matrix
            dissimil_mat = grads @ grads.T  # (N, N)
            np.fill_diagonal(dissimil_mat, 0.0)
            v = - np.sum(dissimil_mat, axis=1) / (N - 1)  # (N,)

            u = eta + v

            # Update bandwith ratio
            R_ratio = np.divide(R, R_ub.reshape((1, -1)))  # (N, G)

        if self.asso_type == 'random':
            conn = []
            for i in range(N):
                avail_ids = np.where(conn_ub[i])[0]
                if self.pref is not None:
                    # Bias random association
                    bias_id = self.labels.index(
                        self.pref[i]) / self.label_num * G
                    idx = (np.abs(bias_id - avail_ids)).argmin()
                    gateway_id = avail_ids[idx]
                elif self.cls_num is not None:
                    # Noniid random association
                    noniid_id = self.cls_num[i]
                    idx = (np.abs(noniid_id - avail_ids)).argmin()
                    gateway_id = avail_ids[idx]
                else:
                    # Pure random association
                    gateway_id = np.random.choice(avail_ids)

                conn.append(np.eye(G)[gateway_id])

            conn = np.array(conn, dtype=np.int)
            print(conn)

        elif self.asso_type == 'gurobi':
            assert grads is not None, "grads should not be none!"

            import gurobipy as gp
            from gurobipy import GRB

            # Create model and variables
            model = gp.Model('clientAssociation')
            model.setParam('MIPGap', 0.01)
            model.setParam('Timelimit', 100)
            vars = model.addMVar(shape=(N, G), vtype=GRB.BINARY, name='vars')
            slack_simil = model.addVar(vtype=GRB.CONTINUOUS, name='slack_simil_var')
            slack_thpt = model.addVar(vtype=GRB.CONTINUOUS, name='slack_thpt_var')

            # Set objective
            model.setObjective(slack_simil - phi * slack_thpt,
                               GRB.MAXIMIZE)

            # Each value appears once per row
            model.addConstrs((sum(vars[i, :]) == 1
                              for i in range(N)), name='unique')

            # Each value is restricted by the feasible connections
            model.addConstrs((vars[:, j] <= conn_ub[:, j] for j in range(G)),
                             name='inclusive')

            # Add constraint for slack similarity
            model.addConstrs((u @ vars[:, j] >= slack_simil
                              for j in range(G)),
                              name='slack_simil')

            # Add constraint for slack delay
            model.addConstrs((R_ratio[:, j] @ vars[:, j] <= slack_thpt
                              for j in range(G)),
                             name='slack_thpt')

            # Optimize model
            model.optimize()

            conn = np.array(vars.X, dtype=np.int)
            # print(conn)
            logging.info('Obj: {}'.format(model.ObjVal))


        else:
            raise ValueError(
                "client association type not implemented: {}".format(self.asso_type))

        logging.info('Obj 1: {}'.format(u @ conn))
        logging.info('Obj 2: {}'.format(np.diag(R_ratio.T @ conn)))

        with open(self.log_file, 'a') as f:
            f.write('Obj 1: {}\n'.format(u @ conn))
            f.write('Obj 2: {}\n'.format(np.diag(R_ratio.T @ conn)))
            for j in range(G):
                f.write('Gw {}: {} clients out of {}\n'.format(
                    j, np.sum(conn[:, j]), np.sum(conn_ub[:, j])
                ))
            f.write('\n')

        return conn
~