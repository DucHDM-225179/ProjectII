import torch
import torch.nn as nn
import torch.nn.functional as F # Often used for activation functions

class DQNetwork(nn.Module):
    """
    Deep Q-Network for Reinforcement Learning.

    This model substitutes the Q-table in Q-Learning. It takes state
    information related to a number of devices and outputs Q-values for
    all possible discrete actions.
    """
    def __init__(self, num_devices: int, hidden_layer_list: list[int] = None):
        """
        Initializes the Deep Q-Network.

        Args:
            num_devices (int): The number of devices (K).
                               Input parameters will be 4 * K.
            hidden_layer_list (list[int], optional):
                               A list of integers, where each integer specifies
                               the number of nodes in that hidden layer.
                               Defaults to [20, 20] (two hidden layers, each with 20 nodes).
        """
        super().__init__() # Call the __init__ of the parent class (nn.Module)

        self.num_devices = num_devices

        # Set default hidden layers if not provided
        if hidden_layer_list is None:
            self.hidden_layer_list = [20, 20]
        else:
            # Ensure it's a list, make a copy to avoid modifying the original if it's passed around
            self.hidden_layer_list = list(hidden_layer_list)

        # Calculate input and output dimensions
        self.input_features = 4 * self.num_devices
        self.output_features = 3 ** self.num_devices # 3^K output nodes

        # --- Construct the layers ---
        layers = []
        current_in_features = self.input_features

        # Add hidden layers
        if self.hidden_layer_list: # Check if the list is not empty
            for num_nodes in self.hidden_layer_list:
                if num_nodes <= 0:
                    raise ValueError("Number of nodes in a hidden layer must be positive.")
                layers.append(nn.Linear(current_in_features, num_nodes))
                layers.append(nn.ReLU()) # Standard activation for hidden layers in DQNs
                current_in_features = num_nodes
        
        # Add the output layer
        # The input to this layer is the output of the last hidden layer,
        # or self.input_features if there are no hidden layers.
        layers.append(nn.Linear(current_in_features, self.output_features))
        # Note: No activation function is typically applied to the final output layer
        # of a DQN, as it directly outputs Q-values which can be any real number.

        # Combine all layers into a sequential model
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the network.

        Args:
            state (torch.Tensor): The input state tensor.
                                  Expected shape: (batch_size, 4 * num_devices)

        Returns:
            torch.Tensor: The Q-values for each action.
                          Shape: (batch_size, 3^num_devices)
        """
        if state.shape[-1] != self.input_features:
            raise ValueError(
                f"Input tensor last dimension ({state.shape[-1]}) "
                f"does not match expected input features ({self.input_features})."
            )
        return self.network(state)

# --- Example Usage ---
if __name__ == '__main__':
    # Scenario 1: Default hidden layers
    num_dev = 2
    model1 = DQNetwork(num_devices=num_dev)
    print(f"Model 1 (num_devices={num_dev}, hidden_layers=default):")
    print(model1)
    print(f"  Input features: {model1.input_features} (4 * {num_dev})")
    print(f"  Output features: {model1.output_features} (3^{num_dev})")

    # Create a dummy batch of input states
    # batch_size = 5, num_devices = 2 => input_features = 4*2 = 8
    dummy_input1 = torch.randn(5, 4 * num_dev)
    output1 = model1(dummy_input1)
    print(f"  Dummy input shape: {dummy_input1.shape}")
    print(f"  Output shape: {output1.shape}")
    print("-" * 30)

    # Scenario 2: Custom hidden layers
    num_dev = 3
    custom_hidden = [64, 32, 16]
    model2 = DQNetwork(num_devices=num_dev, hidden_layer_list=custom_hidden)
    print(f"Model 2 (num_devices={num_dev}, hidden_layers={custom_hidden}):")
    print(model2)
    print(f"  Input features: {model2.input_features} (4 * {num_dev})")
    print(f"  Output features: {model2.output_features} (3^{num_dev})")

    dummy_input2 = torch.randn(10, 4 * num_dev) # batch_size = 10
    output2 = model2(dummy_input2)
    print(f"  Dummy input shape: {dummy_input2.shape}")
    print(f"  Output shape: {output2.shape}")
    print("-" * 30)

    # Scenario 3: No hidden layers
    num_dev = 1
    model3 = DQNetwork(num_devices=num_dev, hidden_layer_list=[]) # Empty list
    print(f"Model 3 (num_devices={num_dev}, hidden_layers=[]):")
    print(model3)
    print(f"  Input features: {model3.input_features} (4 * {num_dev})")
    print(f"  Output features: {model3.output_features} (3^{num_dev})")

    dummy_input3 = torch.randn(2, 4 * num_dev) # batch_size = 2
    output3 = model3(dummy_input3)
    print(f"  Dummy input shape: {dummy_input3.shape}")
    print(f"  Output shape: {output3.shape}")
    print("-" * 30)

    # Example of incorrect input shape
    try:
        num_dev_test = 2
        model_test = DQNetwork(num_devices=num_dev_test)
        wrong_input = torch.randn(1, 4 * num_dev_test + 1) # One extra feature
        model_test(wrong_input)
    except ValueError as e:
        print(f"Caught expected error for wrong input: {e}")