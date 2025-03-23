import matplotlib
matplotlib.use('Qt5Agg') # for command line plotting
import matplotlib.pyplot as plt
plt.ion() # turn on interactive mode
import get_photons
import fitting
import plotting as _p

def scatter_event(i):
    this_event = _p.group_events.iloc[i]

    this_event_photons = get_photons.crop_event(this_event, _p.pick_photons, _p.diameter)
    print(this_event_photons)

    prev_x = this_event.x
    prev_y = this_event.y
    x_array = this_event_photons['x'].to_numpy()
    y_array = this_event_photons['y'].to_numpy()
    new_x, new_y, sx, sy = fitting.event_position(x_array, y_array, False)
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(this_event_photons['x'],
                          this_event_photons['y'],
                          c=this_event_photons['ms'],
                          cmap='viridis')
    #cbar = plt.colorbar(scatter)
    #cbar.set_label('ms value', rotation=270, labelpad=15)
    plt.plot(prev_x, prev_y, 'o', label=f'Prev: ({prev_x:.4f}, {prev_y:.4f})', color='blue')
    plt.plot(new_x, new_y, '^', label=f'New Pos: ({new_x:.4f}, {new_y:.4f})', color='red')
    plt.plot([], [], ' ', label=f'Total number of photons: {len(this_event_photons)}')
    plt.title("Scatter Plot of x, y Positions from DataFrame")
    plt.title('Data Points with Legend')
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.yscale("log")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper left')  # Adjust the legend position if needed

if __name__ == "__main__":
    # Plot the histogram for the first event in _p.group_events
    plt.ioff()
    scatter_event(1)
    plt.show(block=True)
