import os
import imageio
images = []
for i in range(1, 257):
    file_path = os.path.join('png', f"sample_process_zhihu", f'visualization_step_{i}.png')
    images.append(imageio.imread(file_path))
imageio.mimsave('./animation.gif', images)