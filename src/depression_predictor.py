import os
import shutil

class StudentDepressionPredictor:
    def ensure_images_dir(self):
        shutil.rmtree('images', ignore_errors=True)
        os.makedirs('images', exist_ok=True)
        print("Created 'images' directory for saving visualizations")
        return os.path.join(os.getcwd(), 'images')

    # Add the rest of your class implementation here
