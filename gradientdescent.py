import numpy as np
import matplotlib.pyplot as plt

#creating synthetic data
def generate_sample_data(n_samples=50, noise_level=0.1, seed=None):
    if seed is not None:
        np.random.seed(seed)
        
    x = np.linspace(-10, 10, n_samples)
    y = 2 * x + 1 + np.random.normal(0, noise_level, n_samples)
    return x, y

class GradientDescentVisualizer:
    def __init__(self, x_data, y_data):
        self.x_data = np.array(x_data)
        self.y_data = np.array(y_data)
        
    def linear_gradient_descent(self, learning_rate=0.001, iterations=100):
        m = 5.0  
        n = len(self.x_data)
        m_history = []
        cost_history = []
        
        for i in range(iterations):
            y_pred = m * self.x_data
            error = self.y_data - y_pred
            gradient = -2 * np.mean((self.y_data - m * self.x_data) * self.x_data) #derived as -2 * mean(error * x)
            m = m - learning_rate * gradient
            cost = np.mean(error ** 2)
            
            m_history.append(m)
            cost_history.append(cost)
            
            if abs(gradient) < 1e-6:
                break
                
        return m, m_history, cost_history

    def plot_optimization_visualizations(self):
        #running gradient descent
        best_m_gd, m_history, cost_history = self.linear_gradient_descent()
        
        #plotting
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        #graph 1: Best m Value
        m_range = np.linspace(0, 5, 100)
        costs = [np.mean((self.y_data - m * self.x_data) ** 2) for m in m_range] #linear
        best_m_ls_idx = np.argmin(costs)
        best_m_ls = m_range[best_m_ls_idx]
        
        #error landscaoe
        axes[0].plot(m_range, costs, 'b-', label='MSE vs m')
        
        #linear search result
        axes[0].axvline(best_m_ls, color='r', linestyle='--', 
                       label=f'Linear Search (m={best_m_ls:.4f})')
        
        #gradient descent result
        axes[0].axvline(best_m_gd, color='g', linestyle=':', 
                       label=f'Gradient Descent (m={best_m_gd:.4f})')
        
        axes[0].set_xlabel('m values')
        axes[0].set_ylabel('MSE')
        axes[0].set_title('Finding Best m Value')
        axes[0].legend()
        axes[0].grid(True)
        
        #graph 2: convergence of gradient descent
        axes[1].plot(cost_history, 'b-')
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Mean Squared Error')
        axes[1].set_title('Gradient Descent Convergence\nfor Linear Regression')
        axes[1].grid(True)
        
        #graph 3:Local minima 
        m_values = np.linspace(0, 5, 100)
        cost_values = [np.mean((self.y_data - m * self.x_data) ** 2) for m in m_values]
        
        min_index = np.argmin(cost_values)
        min_m = m_values[min_index]  #lowest cost m
        
        axes[2].plot(m_values, cost_values, 'b-', label='MSE Cost Function')
        axes[2].scatter([min_m], [cost_values[min_index]], color='red', s=100, marker='*',
                        label=f'Local Minimum (m={min_m:.4f})')
        axes[2].set_xlabel('m values')
        axes[2].set_ylabel('Mean Squared Error')
        axes[2].set_title('Finding Local Minima using Gradient Descent')
        axes[2].legend()
        axes[2].grid(True)


if __name__ == "__main__":
    # Generate and visualize linear data
    x_linear, y_linear = generate_sample_data(seed=42)
    visualizer = GradientDescentVisualizer(x_linear, y_linear)
    visualizer.plot_optimization_visualizations()
