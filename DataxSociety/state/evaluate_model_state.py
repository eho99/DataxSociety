"""
State to fetch access to model and data a user has used from a database
"""

from sqlmodel import select

import reflex as rx

from ..models.project import *
from ..models.profile_project_link import *
from ..models.project_models import *
from ..models.project_data import *

from .login_state import LOGIN_ROUTE, REGISTER_ROUTE
from .base_state import State
from ..backend.DENSE import * 
from ..backend.LSTM import * 

class Eval_State(State):
    """Return the accuracy of the model on the given dataset query

    Returns: int representing the accuracy of the model
        """

    def eval_model(self, project_id, model_id):
        
        with rx.session as session:
            query_result = (
                session.query(Project, ProjectModel)
                .join(ProjectModel)
                .filter(Project.id == project_id)
                .filter(ProjectModel.id == model_id)
                .first()
            )

            proj, proj_model = query_result

            input_nodes = proj.num_indep_vars
            output_nodes = proj.num_dep_vars

            data_table_id = proj.project_data
            data_set, label_set = (session.query(ProjectData).filter(ProjectData.data_table_id == data_table_id).first())


            # hyperparams
            layer_nodes = proj_model.layer_nodes
            layer_activations = proj_model.layer_activations
            learning_rate = proj_model.learning_rate
            num_epochs = proj_model.num_epochs
            test_ratio = proj_model.test_ratio
            loss = proj_model.loss_func

            # I expect this to be LSTM or DENSE
            if proj_model.network_type == "DENSE":
                network = DENSE(input_nodes, output_nodes, layer_nodes, layer_activations, [i for i in range(output_nodes)])
                test_loss, epoch_status, test_acuracy = network.train(loss, learning_rate, num_epochs, test_ratio, data_set, label_set)
                return test_acuracy
            elif proj_model.network_type == "LSTM":
                pass





