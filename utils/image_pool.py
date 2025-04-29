import random
import torch


class ImagePool:
    """This class implements an image buffer that stores previously generated images.
    
    This buffer enables updating discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class.
        
        Args:
            pool_size (int): The size of image buffer, if pool_size=0, no buffer will be created.
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return images from the pool.
        
        By 50/100, the function returns the input images.
        By 50/100, the function returns images previously stored in the pool,
        and inserts the current images into the pool.
        
        Args:
            images (torch.Tensor): The latest generated images from the generator.
            
        Returns:
            torch.Tensor: Images from the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                # If the buffer is not full, keep inserting current images into the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                # If the buffer is full, by 50% chance, replace a random image in the buffer with the current one
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    # By another 50% chance, return the current image itself
                    return_images.append(image)
        
        # Collect all return images and return them as a single tensor
        return_images = torch.cat(return_images, 0)
        return return_images 