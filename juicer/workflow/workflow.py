import networkx as nx
import matplotlib.pyplot as plt
from juicer.service import tahiti_service

class Workflow:
    """
        - Set and get Create a graph
        - Identify tasks and flows
        - Set and get workflow
        - Add edges between tasks (source_id and targed_id)

    """
    WORKFLOW_DATA_PARAM = 'workflow_data'
    WORKFLOW_GRAPH_PARAM = 'workflow_graph'
    WORKFLOW_GRAPH_SORTED_PARAM = 'workflow_graph_sorted'
    WORKFLOW_PARAM = 'workflow'
    GRAPH_PARAM = 'graph'

    WORKFLOW_GRAPH_SOURCE_ID_PARAM = 'source_id'
    WORKFLOW_GRAPH_TARGET_ID_PARAM = 'target_id'

    def __init__(self, workflow_data):

        # Initialize
        self.workflow_graph = nx.MultiDiGraph()

        # Workflow dictionary
        self.workflow_data = workflow_data

        # Construct graph
        self.workflow_graph = self.builds_initial_workflow_graph()

        # Topological sorted tasks according to their dependencies
        self.sorted_tasks = []

        # Verify null edges to topological_sorted_tasks
        if self.is_there_null_target_id_tasks() \
                and \
            self.is_there_null_source_id_tasks():

            self.sorted_tasks = self.get_topological_sorted_tasks()
        else:
            raise AttributeError(
                "Port '{}/{}' must be informed for operation{}".format(
                    self.WORKFLOW_GRAPH_SOURCE_ID_PARAM,
                    self.WORKFLOW_GRAPH_TARGET_ID_PARAM,
                    self.__class__))



    def builds_initial_workflow_graph(self):
        """ Builds a graph with the tasks """

        for task in self.workflow_data['tasks']:

            query_tahiti = self.get_ports_from_operation_tasks(
                                    task.get('operation')['id'])

            self.workflow_graph.add_node(
                task.get('id'),
                in_degree_required=query_tahiti['N_INPUT'],
                in_degree_multiplicity_required=query_tahiti['M_INPUT'],
                out_degree_required=query_tahiti['N_OUTPUT'],
                ou_degree_multiplicity_required=query_tahiti['M_OUTPUT'],
                attr_dict=task)

        for flow in self.workflow_data['flows']:
            self.workflow_graph.add_edge(flow['source_id'], flow['target_id'],
                                attr_dict=flow)

        for nodes in self.workflow_graph.nodes():
            self.workflow_graph.node[nodes]['in_degree'] = self.workflow_graph.\
                in_degree(nodes)

            self.workflow_graph.node[nodes]['out_degree'] = self.workflow_graph.\
                out_degree(nodes)

        return self.workflow_graph


    def check_in_degree_edges(self):
        for nodes in self.workflow_graph.nodes():
            if self.workflow_graph.node[nodes]['in_degree'] == \
                    self.workflow_graph.node[nodes]['in_degree_required']:
                pass
            else:
                raise AttributeError(
                    "Port '{} in node {}' missing, must be informed for operation {}".
                    format(
                        self.WORKFLOW_GRAPH_TARGET_ID_PARAM,
                        nodes,
                        self.__class__))
        return 1

    def check_out_degree_edges(self):

        for nodes in self.workflow_graph.nodes():
            if self.workflow_graph.node[nodes]['out_degree'] == \
                    self.workflow_graph.node[nodes]['out_degree_required']:
                pass
            else:
                raise AttributeError(

                    "Port '{}' missing, must be informed for operation {}".format(
                        self.WORKFLOW_GRAPH_SOURCE_ID_PARAM,
                        self.__class__)
                )
        return 1


    def builds_sorted_workflow_graph(self, tasks, flows):

        workflow_graph = nx.MultiDiGraph()

        for task in tasks:
            query_tahiti = self.get_ports_from_operation_tasks(
                                    task.get('operation')['id'])

            workflow_graph.add_node(
                task.get('id'),
                in_degree_required=query_tahiti['N_INPUT'],
                in_degree_multiplicity_required=query_tahiti['M_INPUT'],
                out_degree_required=query_tahiti['N_OUTPUT'],
                ou_degree_multiplicity_required=query_tahiti['M_OUTPUT'],
                attr_dict=task)

        for flow in flows:
            workflow_graph.add_edge(flow['source_id'], flow['target_id'],
                                         attr_dict=flow)

        print self.workflow_graph.nodes()
        # updating in_degree and out_degree
        for nodes in self.workflow_graph.nodes():
            self.workflow_graph.node[nodes]['in_degree'] = self.workflow_graph. \
                in_degree(nodes)
            self.workflow_graph.node[nodes]['out_degree'] = self.workflow_graph. \
                out_degree(nodes)
        return workflow_graph


    def plot_workflow_graph_image(self):
        """
             Show the image from workflow_graph
        """
        # Change layout according to necessity
        pos = nx.spring_layout(self.workflow_graph)
        nx.draw(self.workflow_graph, pos, node_color='#004a7b', node_size=2000,
                edge_color='#555555', width=1.5, edge_cmap=None,
                with_labels=True, style='dashed',
                label_pos=50.3, alpha=1, arrows=True, node_shape='s',
                font_size=8,
                font_color='#FFFFFF')
        plt.show()
        # If necessary save the image
        # plt.savefig(filename, dpi=300, orientation='landscape', format=None,
        # bbox_inches=None, pad_inches=0.1)

    def get_topological_sorted_tasks(self):

        """ Create the tasks Graph and perform topological sorting

            A topological sort is a nonunique permutation of the nodes
            such that an edge from u to v implies that u appears before
            v in the topological sort order.

            :return: Return a list of nodes in topological sort order.
        """
        # First, map the tasks IDs to their original position
        tasks_position = {}

        for count_position, task in enumerate(self.workflow_data['tasks']):
            tasks_position[task['id']] = count_position

        sorted_tasks_id = nx.topological_sort(self.workflow_graph, reverse=False)

        return sorted_tasks_id

    def is_there_null_source_id_tasks(self):
        for flow in self.workflow_data['flows']:
            if flow['source_id'] == "":
                return False
        return True

    def is_there_null_target_id_tasks(self):
        for flow in self.workflow_data['flows']:
            if flow['target_id'] == "":
                return False
        return True


    # @FIX-ME - NOT WORKING YET
    def get_all_ports_operations_tasks(self):
        params = {
            'base_url':'http://beta.ctweb.inweb.org.br',
            'item_path': 'tahiti/operations',
            'token': '123456',
            'item_id': ''
        }

        # Querying tahiti operations to get number of inputs and outputs
        operations = tahiti_service.query_tahiti(params['base_url'],
                                                       params['item_path'],
                                                       params['token'],
                                                       params['item_id'])
        return operations

    def get_ports_from_operation_tasks(self, id_operation):
        # Can i put this information here?
        params = {
            'base_url':'http://beta.ctweb.inweb.org.br',
            'item_path': 'tahiti/operations',
            'token': '123456',
            'item_id': id_operation
        }

        # Querying tahiti operations to get number of inputs and outputs
        operations_ports = tahiti_service.query_tahiti(params['base_url'],
                                                   params['item_path'],
                                                   params['token'],
                                                   params['item_id'])
        # Get operation requirements in tahiti
        result = {
                    'N_INPUT': 0,
                    'N_OUTPUT': 0,
                    'M_INPUT': 'None',
                    'M_OUTPUT': 'None'
                 }

        for port in operations_ports['ports']:
            if port['type'] == 'INPUT':
                result['M_INPUT'] = port['multiplicity']
                if 'N_INPUT' in result:
                    result['N_INPUT'] += 1
                else:
                    result['N_INPUT'] = 1
            elif port['type'] == 'OUTPUT':
                result['M_OUTPUT'] = port['multiplicity']
                if 'N_OUTPUT' in result:
                    result['N_OUTPUT'] += 1
                else:
                    result['N_OUTPUT'] = 1
        return result

    def workflow_execution_parcial(self):

        topological_sort = self.get_topological_sorted_tasks()

        for node_obj in topological_sort:
            # print self.workflow_graph.node[node]
            print (nx.ancestors(self.workflow_graph,node_obj),
                   self.workflow_graph.predecessors(node_obj),
                   node_obj,
                   self.workflow_graph.node[node_obj]['in_degree_required'],
                   self.workflow_graph.node[node_obj]['in_degree'],
                   self.workflow_graph.node[node_obj]['out_degree_required'],
                   self.workflow_graph.node[node_obj]['out_degree']
                   )
        print topological_sort

        return True
    # only to debug
    def check_outdegree_edges(self, atr):

        if self.workflow_graph.has_node(atr):
            return (self.workflow_graph.node[atr]['in_degree'],
                    self.workflow_graph.node[atr]['out_degree'],
                    self.workflow_graph.in_degree(atr),
                    self.workflow_graph.out_degree(atr),
                    self.workflow_graph.node[atr]['in_degree_required'],
                    self.workflow_graph.node[atr]['out_degree_required']
                    )
        else:
            raise KeyError("The node informed doesn't exist")
        # return x
   # def verify_workflow(self):
   #     """
    #     Verifies if the workflow is valid.
    #     Validations to be implemented:
    #     - Supported platform
    #     - Workflow without input
    #     - Supported operations in platform
    #     - Consistency between tasks and flows
    #     - Port consistency
    #     - Task parameters
    #     - Referenced attributes names existing in input dataframes
    #     """
    #     pass

        # def sort_tasks(self):
        #     """ Create the tasks Graph and perform topological sorting """
        #     # First, map the tasks IDs to their original position
        #     tasks_position = {}
        #
        #     for count_position, task in enumerate(self.workflow['tasks']):
        #         tasks_position[task['id']] = count_position
        #
        #     # Then, performs topological sorting
        #     workflow_graph = self.builds_workflow_graph(
        #         self.workflow['tasks'], self.workflow['flows'])
        #     sorted_tasks_id = nx.topological_sort(workflow_graph, reverse=False)
        #     # Finally, create a new array of tasks in the topogical order
        #     for task_id in sorted_tasks_id:
        #         self.sorted_tasks.append(
        #             self.workflow['tasks'][tasks_position[task_id]])

        # def plot_workflow(self, filename):
        #    """ Plot the workflow graph """
        # workflow_graph = self.builds_workflow_graph(self.sorted_tasks,
        #                                                self.workflow['flows'])
        # pos = nx.spring_layout(workflow_graph)
        # nx.draw(workflow_graph, pos, node_color='#004a7b', node_size=2000,
        #         edge_color='#555555', width=1.5, edge_cmap=None,
        #         with_labels=True,
        #         label_pos=50.3, alpha=1, arrows=True, node_shape='s',
        #         font_size=8,
        #         font_color='#FFFFFF')
        # plt.savefig(filename, dpi=300, orientation='landscape', format=None,
        #             bbox_inches=None, pad_inches=0.1)

        # def builds_workflow_graph(self, tasks, flows):
        #     """ Builds a graph with the tasks """
        #     workflow_graph = nx.DiGraph()
        #
        #     for task in tasks:
        #         workflow_graph.add_node(task['id'])
        #
        #     for flow in flows:
        #         workflow_graph.add_edge(flow['source_id'], flow['target_id'])
        #     return workflow_graph
