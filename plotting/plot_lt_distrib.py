import os
import glob
import matplotlib.pyplot as plt
from utilities import helper


def process_folder(folder):
    """
    Processes all .hdf5 files in the specified folder.
    For each file, it uses helper.process_input() to load the data as a DataFrame,
    then calculates the average and standard deviation of the 'lifetime' column.

    Returns:
        avg_lifetimes (list): List of average lifetime values.
        std_lifetimes (list): List of lifetime standard deviation values.
    """
    # Get all .hdf5 files in the folder and sort them (assumes filenames sort in the desired order)
    file_list = sorted(glob.glob(os.path.join(folder, '*.hdf5')))

    avg_lifetimes = []
    std_lifetimes = []

    for f in file_list:
        # Process the file to get a DataFrame
        df = helper.process_input(f, 'locs')
        # Calculate average and standard deviation for the 'lifetime' column
        avg_lifetimes.append(df['lifetime'].mean())
        std_lifetimes.append(df['lifetime'].std())

    return avg_lifetimes, std_lifetimes


def main():
    # Define the folders to process
    folders = ['a550', 'a565']

    # Dictionary to store results for each folder
    results = {}

    for folder in folders:
        avg, std = process_folder(folder)
        # Compute normalized std: std/avg. Avoid division by zero.
        norm_std = [s / a if a != 0 else 0 for s, a in zip(std, avg)]
        results[folder] = {'avg': avg, 'std': std, 'norm_std': norm_std}

    # Create an x-axis corresponding to file number (assumes same number of files per folder)
    file_numbers = [10, 15, 20, 25]#range(1, len(results[folders[0]]['avg']) + 1)

    # Plot 1: Average Lifetime vs. File Number
    plt.figure(figsize=(8, 6))
    for folder in folders:
        plt.plot(file_numbers, results[folder]['avg'], marker='o', label=folder)
    plt.xlabel('decay time [ns]')
    plt.ylabel('Average lifetime')
    plt.title('Average lifetime vs. decay time')
    plt.xticks(file_numbers)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot 2: Normalized Standard Deviation vs. File Number
    plt.figure(figsize=(8, 6))
    for folder in folders:
        plt.plot(file_numbers, results[folder]['norm_std'], marker='o', label=f"Normalized std {folder}")
    plt.xlabel('decay time')
    plt.ylabel('Normalized Standard Deviation (std/avg)')
    plt.title('Normalized Standard Deviation vs. decay time')
    plt.xticks(file_numbers)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot 3: Combined Plot
    # Calculate the absolute difference between average lifetimes of a550 and a565.
    avg_diff = [abs(a550 - a565) for a550, a565 in zip(results['a550']['avg'], results['a565']['avg'])]

    plt.figure(figsize=(8, 6))
    plt.plot(file_numbers, avg_diff, marker='o', linestyle='--', label='Avg lifetime Difference (|a550 - a565|)')
    plt.plot(file_numbers, results['a550']['std'], marker='o', label='std a550')
    plt.plot(file_numbers, results['a565']['std'], marker='o', label='std a565')
    plt.xlabel('Decay time window [ns]')
    plt.ylabel('Value')
    plt.title('Avg lifetime Difference and Std vs. Decay time window, min 200 photons')
    plt.xticks(file_numbers)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()