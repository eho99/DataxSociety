"""
State to fetch access to model and data a user has used from a database
"""

from sqlmodel import select

import reflex as rx

from ..models.project import *
from ..models.profile_project_link import *
from ..models.mnistnetwork import *
from ..models.mnistdata import *
from ..models.user import User
from ..models.mnistnetwork import MNISTNetwork

from .login_state import LOGIN_ROUTE, REGISTER_ROUTE
from .base_state import State
from ..backend.DENSE import DENSE
from ..backend.LSTM import LSTM

class EvalModelState(State):
    """Return the accuracy of the model on the given dataset query

    Returns: int representing the accuracy of the model
        """
    def on_submit(self, form_data) -> rx.event.EventSpec:
        project_name = form_data["project_name"]
        model_type = form_data["model_type"]
        num_layers = form_data["num_layers"]
        hidden_size = form_data["hidden_size"]
        activations = form_data["activations"]
        dropout = form_data["dropout"]
        epochs = form_data["epochs"]
        test_ratio = form_data["test_ratio"]
        learning_rate = form_data["learning_rate"]
        loss_func = form_data["loss_function"]

        creator = form_data["username"]

        with rx.session() as session:
            result = session.exec(
                session.query(User).where(User.username == creator)
            ).first()
            creator_id = result[0].id

        network = MNISTNetwork()
        network.creator_id = creator_id
        network.num_hidden_layers = num_layers
        network.layer_nodes = hidden_size
        network.layer_activations = activations
        network.learning_rate = learning_rate
        network.loss_func = loss_func
        network.test_ratio = test_ratio
        network.network_type = model_type
        network.dropout = dropout

        with rx.session() as session:
            session.add(network)
            session.commit()

            # FIX THIS!!!!
            
            result = session.exec(
                session.query(MNISTNetwork).filter(MNISTNetwork.creator_id == creator_id).order_by(MNISTNetwork.id.desc())
                ).first()
            network_id = result[0].id

        accuracy = self.eval_model(1, network_id)
        
        return EvalModelState.redir()  

    def redir(self) -> rx.event.EventSpec: 
        return rx.redirect("/train_model")

    def eval_model(self, project_id, model_id):
        with rx.session() as session:
            query_result = (
                session.query(Project, MNISTNetwork)
                # .join(MNISTNetwork)
                .filter(Project.id == project_id)
                .filter(MNISTNetwork.id == model_id)
            ).first()

            print(query_result)

            proj, proj_model = query_result

            input_nodes = proj.num_indep_vars
            output_nodes = proj.num_dep_vars

            # data_table_id = proj.project_data
            # data_set, label_set = session.query(MNISTData).filter(MNISTData.data_table_id == data_table_id).first()

            # data_query = (
            #     session.query(MNISTData)
            # ).all()
            # print(data_query)
            # data_set, label_set = data_query.pixel_vals, data_query.label

            data_set = (
                session.query(MNISTData.pixel_vals)
            ).all()

            label_set = (
                session.query(MNISTData.label)
            ).all()


            # hyperparams
            layer_nodes = proj_model.layer_nodes
            layer_activations = proj_model.layer_activations
            learning_rate = proj_model.learning_rate
            num_epochs = proj_model.num_epochs
            test_ratio = proj_model.test_ratio
            loss = proj_model.loss_func
            dropout = proj_model.dropout

            print(proj_model.network_type)

            # I expect this t so be LSTM or DENSE
            if proj_model.network_type == 'Dense':
                print(layer_nodes, layer_activations)
                network = DENSE(input_nodes, output_nodes, layer_nodes, layer_activations, [i for i in range(output_nodes)])
                
                test_loss, epoch_status, test_acuracy = network.train(loss, learning_rate, num_epochs, test_ratio, data_set, label_set)
                print(test_acuracy)
                proj_model.accuracy = test_acuracy
                return test_acuracy
            elif proj_model.network_type == "LSTM":
                network = DENSE(input_nodes, output_nodes, layer_nodes, layer_nodes+2, dropout)
                test_loss, epoch_status, test_acuracy = network.train(loss, learning_rate, num_epochs, test_ratio, data_set, label_set)
                return test_acuracy




