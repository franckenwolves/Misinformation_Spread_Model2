import math
import sys
import mesa
import socket
import errno
from .model import VirusOnNetwork#, State#, number_infected


def network_portrayal(G):
    # The model ensures there is always 1 agent per node

    def node_color(agent, virus):
        for i in agent.misinformation:
            if i < agent.misinformation[0]['num_virus']:
                if agent.misinformation[i]['infected'] == 'yes':
                    if i == 0:
                        return "#ff0000"
                    if i == 1:
                        return "#00ff00"
                    if i == 2:
                        return "#0000ff"

        '''if agent.misinformation[0]['exposed'] == 'no' and agent.misinformation[1]['exposed'] == 'no' and agent.misinformation[2]['exposed'] == 'no':
            return "#008000"
        elif agent.misinformation[0]['exposed'] == 'yes' and agent.misinformation[1]['exposed'] == 'no' and agent.misinformation[2]['exposed'] == 'no':
            return "#FFFC33"
        elif agent.misinformation[0]['exposed'] == 'no' and agent.misinformation[1]['exposed'] == 'yes' and agent.misinformation[2]['exposed'] == 'no':
            return "#FFFC33"
        elif agent.misinformation[0]['exposed'] == 'no' and agent.misinformation[1]['exposed'] == 'no' and agent.misinformation[2]['exposed'] == 'yes':
            return "#DF5FED"
        elif agent.misinformation[0]['exposed'] == 'yes' and agent.misinformation[1]['exposed'] == 'yes' and agent.misinformation[2]['exposed'] == 'no':
            return "#4A2804"
        elif agent.misinformation[0]['exposed'] == 'yes' and agent.misinformation[1]['exposed'] == 'no' and agent.misinformation[2]['exposed'] == 'yes':
            return "#4A2804"
        elif agent.misinformation[0]['exposed'] == 'no' and agent.misinformation[1]['exposed'] == 'yes' and agent.misinformation[2]['exposed'] == 'yes':
            return "#4A2804"
        elif agent.misinformation[0]['exposed'] == 'yes' and agent.misinformation[1]['exposed'] == 'yes' and agent.misinformation[2]['exposed'] == 'yes':
            return "#4A2804"
        elif agent.misinformation[0]['infected'] == 'yes' and agent.misinformation[1]['exposed'] == 'no' and agent.misinformation[2]['exposed'] == 'no':
            return "#FF0000"
        elif agent.misinformation[0]['infected'] == 'yes' and agent.misinformation[1]['exposed'] == 'yes' and agent.misinformation[2]['exposed'] == 'no':
            return "#0D1FDE"
        elif agent.misinformation[0]['infected'] == 'yes' and agent.misinformation[1]['exposed'] == 'no' and agent.misinformation[2]['exposed'] == 'yes':
            return "#0D1FDE"
        elif agent.misinformation[0]['infected'] == 'yes' and agent.misinformation[1]['exposed'] == 'no' and agent.misinformation[2]['infected'] == 'yes':
            return "#0D1FDE"
        elif agent.misinformation[0]['infected'] == 'yes' and agent.misinformation[1]['exposed'] == 'yes' and agent.misinformation[2]['infected'] == 'yes':
            return "#0D1FDE"
        elif agent.misinformation[0]['exposed'] == 'yes' and agent.misinformation[1]['infected'] == 'yes' and agent.misinformation[2]['exposed'] == 'no':
            return "#720F7D"
        elif agent.misinformation[0]['exposed'] == 'no' and agent.misinformation[1]['infected'] == 'yes' and agent.misinformation[2]['exposed'] == 'yes':
            return "#720F7D"
        elif agent.misinformation[0]['exposed'] == 'no' and agent.misinformation[1]['infected'] == 'yes' and agent.misinformation[2]['exposed'] == 'yes':
            return "#720F7D"
        elif agent.misinformation[0]['exposed'] == 'no' and agent.misinformation[1]['infected'] == 'yes' and agent.misinformation[2]['infected'] == 'yes':
            return "#720F7D"
        elif agent.misinformation[0]['exposed'] == 'yes' and agent.misinformation[1]['infected'] == 'yes' and agent.misinformation[2]['infected'] == 'yes':
            return "#720F7D"
        elif agent.misinformation[0]['exposed'] == 'yes' and agent.misinformation[1]['exposed'] == 'yes' and agent.misinformation[2]['infected'] == 'yes':
            return "#720F7D"
        elif agent.misinformation[0]['exposed'] == 'yes' and agent.misinformation[1]['exposed'] == 'no' and agent.misinformation[2]['infected'] == 'yes':
            return "#720F7D"
        elif agent.misinformation[0]['exposed'] == 'no' and agent.misinformation[1]['exposed'] == 'yes' and agent.misinformation[2]['infected'] == 'yes':
            return "#720F7D"'''

        '''
        if agent.misinformation[virus]['exposed'] == 'no':
            return "#008000"
        elif agent.misinformation[virus]['exposed'] == 'yes' and agent.misinformation[virus]['infected'] == 'no':
            return "#FFFF00"
        elif agent.misinformation[virus]['exposed'] == 'yes' and agent.misinformation[virus]['infected'] == 'yes':
            return "#FF0000"'''




    def edge_color(agent1, agent2):
        if agent1.misinformation[0]['skeptical_level'] == 1 and agent1.misinformation[1]['skeptical_level'] == 1 and agent2.misinformation[0]['skeptical_level'] == 1 and agent2.misinformation[1]['skeptical_level'] == 1:
            return "#000000"
        return "#e8e8e8"

    def edge_width(agent1, agent2):
        if agent1.misinformation[0]['skeptical_level'] == 1 and agent1.misinformation[1]['skeptical_level'] == 1 and agent2.misinformation[0]['skeptical_level'] == 1 and agent2.misinformation[1]['skeptical_level'] == 1:
            return 3
        return 2

    def get_agents(source, target):
        return G.nodes[source]["agent"][0], G.nodes[target]["agent"][0],

    portrayal = dict()
    portrayal["nodes"] = [
        {
            "size": 6,
            "color": node_color(agents[0], agents[0].virus),
            "tooltip": f"id: {agents[0].unique_id}<br>Target nodes: {G[_]} "
                       #f"{'exposed virus 1:', agents[0].misinformation[0]['exposed'],'infected virus 1:', agents[0].misinformation[0]['infected']}"
                       #f"{'exposed virus 2:', agents[0].misinformation[1]['exposed'],'infected virus 2:', agents[0].misinformation[1]['infected']}"
                       f"<br> Virus 0: {agents[0].misinformation[0]}"
                       f"<br> Virus 1: {agents[0].misinformation[1]}"
                       f"<br> Virus 2: {agents[0].misinformation[2]}",
        }
        for (_, agents) in G.nodes.data("agent")
    ]

    portrayal["edges"] = [
        {
            "source": source,
            "target": target,
            "color": edge_color(*get_agents(source, target)),
            "width": edge_width(*get_agents(source, target)),
        }
        for (source, target) in G.edges
    ]

    return portrayal


network = mesa.visualization.NetworkModule(network_portrayal, 750, 750)
chart = mesa.visualization.ChartModule(
    [
        {"Label": "Susceptible", "Color": "#008000"},
        {"Label": "Exposed 0 and Susceptible 1", "Color": "#FFFC33"},
        {"Label": "Exposed 1 and Susceptible 0", "Color": "#DF5FED"},
        {"Label": "Exposed Both", "Color": "#4A2804"},
        {"Label": "Infected 0 Susceptible 1", "Color": "#FF0000" },
        {"Label": "Infected 1 Susceptible 0", "Color": "#0D1FDE"},
        {"Label": "Infected 0 Exposed 1", "Color": "#E66C20"},
        {"Label": "Infected 1 Exposed 0", "Color": "#720F7D"},
        {"Label": "Skeptical", "Color": "#808080"},
    ]
)


def get_skeptical_susceptible_ratio(model):
    ratio = model.skeptical_susceptible_ratio()
    ratio_text = "&infin;" if ratio is math.inf else f"{ratio:.2f}"
    infected_text = str(number_infected(model))

    return "Skeptical/Susceptible Ratio: {}<br>Infected Remaining: {}".format(
        ratio_text, infected_text
    )


model_params = {
    "j": mesa.visualization.Slider(
        "Choose number of viruses",
        2,   #starting value
        1,   #beginning value of scale
        3,   #ending value of scale
        1,   #interval by which scale is increased
        description="Choose the number of viruses you would like the model to run",
    ),
    "num_nodes": mesa.visualization.Slider(
        "Number of agents",
        50,   #starting value
        10,   #beginning value of scale
        100,   #ending value of scale
        1,   #interval by which scale is increased
        description="Choose how many agents to include in the model",
    ),
    "avg_node_degree": mesa.visualization.Slider(
        "Avg Node Degree", 3, 3, 8, 1, description="Avg Node Degree"
    ),
    "initial_outbreak_size_virus_0": mesa.visualization.Slider(
        "Initial Outbreak Size Virus 0",
        1,
        1,
        10,
        1,
        description="Initial Outbreak Size of virus 0",
    ),
    "initial_outbreak_size_virus_1": mesa.visualization.Slider(
        "Initial Outbreak Size Virus 1",
        1,
        1,
        10,
        1,
        description="Initial Outbreak Size of virus 1",
    ),
    "initial_outbreak_size_virus_2": mesa.visualization.Slider(
        "Initial Outbreak Size Virus 2",
        1,
        1,
        10,
        1,
        description="Initial Outbreak Size of virus 2",
    ),
    "virus_0_spread_chance": mesa.visualization.Slider(
        "Virus 0 Spread Chance",
        0.4,
        0.0,
        1.0,
        0.1,
        description="Probability that susceptible neighbor will be infected with virus 0",
    ),
    "virus_1_spread_chance": mesa.visualization.Slider(
        "Virus 1 Spread Chance",
        0.4,
        0.0,
        1.0,
        0.1,
        description="Probability that susceptible neighbor will be infected with virus 1",
    ),
    "virus_2_spread_chance": mesa.visualization.Slider(
        "Virus 2 Spread Chance",
        0.4,
        0.0,
        1.0,
        0.1,
        description="Probability that susceptible neighbor will be infected with virus 2",
    ),
    "virus_0_check_frequency": mesa.visualization.Slider(
        "Virus 0 Check Frequency",
        0.4,
        0.0,
        1.0,
        0.1,
        description="Frequency the nodes check whether they are infected by " "a virus",
    ),
    "virus_1_check_frequency": mesa.visualization.Slider(
        "Virus 1 Check Frequency",
        0.4,
        0.0,
        1.0,
        0.1,
        description="Frequency the nodes check whether they are infected by " "a virus",
    ),
    "virus_2_check_frequency": mesa.visualization.Slider(
        "Virus 2 Check Frequency",
        0.4,
        0.0,
        1.0,
        0.1,
        description="Frequency the nodes check whether they are infected by " "a virus",
    ),
    "exposed_chance_virus_0": mesa.visualization.Slider(
        "Exposed Chance Virus 0",
        0.3,
        0.0,
        1.0,
        0.1,
        description="Probability that the individual will be exposed to this virus",
    ),
    "exposed_chance_virus_1": mesa.visualization.Slider(
        "Exposed Chance Virus 1",
        0.3,
        0.0,
        1.0,
        0.1,
        description="Probability that the individual will be exposed to this virus",
    ),
    "exposed_chance_virus_2": mesa.visualization.Slider(
        "Exposed Chance Virus 2",
        0.3,
        0.0,
        1.0,
        0.1,
        description="Probability that the individual will be exposed to this virus",
    ),
    "gain_skeptical_chance_virus_0": mesa.visualization.Slider(
        "Gain Skeptical Chance",
        0.5,
        0.0,
        1.0,
        0.1,
        description="Probability that a recovered agent will become "
        "skeptical to this virus in the future",
    ),
    "gain_skeptical_chance_virus_1": mesa.visualization.Slider(
        "Gain Skeptical Chance",
        0.5,
        0.0,
        1.0,
        0.1,
        description="Probability that a recovered agent will become "
        "skeptical to this virus in the future",
    ),
    "gain_skeptical_chance_virus_2": mesa.visualization.Slider(
        "Gain Skeptical Chance",
        0.5,
        0.0,
        1.0,
        0.1,
        description="Probability that a recovered agent will become "
        "skeptical to this virus in the future",
    ),
}

server = mesa.visualization.ModularServer(
    VirusOnNetwork,
    [network, chart], #need to add ", get_skeptical_susceptible_ratio" to this line in between network and charty to get the original
    "Virus Model",
    model_params,
)