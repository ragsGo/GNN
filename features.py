from load import load_data
from main import get_or_create
import matplotlib.pyplot as plt
import numpy as np

from captum.attr import IntegratedGradients
from captum.attr import LayerConductance
from captum.attr import NeuronConductance


# Helper method to print importances and visualize distribution
def visualize_importances(feature_names, importances, title="Average Feature Importances", plot=True,
                          axis_title="Features"):
    print(title)
    for i in range(len(feature_names)):
        print(feature_names[i], ": ", '%.3f' % (importances[i]))
    x_pos = (np.arange(len(feature_names)))
    if plot:
        plt.figure(figsize=(12, 6))
        plt.bar(x_pos, importances, align='center')
        plt.xticks(x_pos, feature_names, wrap=True)
        plt.xlabel(axis_title)
        plt.title(title)


if __name__ == "__main__":
    # model = create_network(num_features, 1)
    dataset = load_data("SNP.csv")
    test_input_tensor = dataset[0].test.x, dataset[0].test.edge_index
    model = get_or_create(1)
    ig = IntegratedGradients(model)
    test_input_tensor[0].requires_grad_()
    print(test_input_tensor)
    attr, delta = ig.attribute(test_input_tensor, target=1, return_convergence_delta=True)
    attr = attr.detach().numpy()

    visualize_importances(list(range(dataset.num_features)), np.mean(attr, axis=0))

