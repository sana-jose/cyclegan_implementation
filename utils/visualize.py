import matplotlib.pyplot as plt
def image_to_tensor(tensor_image): 
    image=tensor_image.cpu().clone().detach().numpy()
    image=image.squeeze(0)
    image=image.transpose(1,2,0)
    image=(image+1)/2
    image=image.clip(0,1)
    image=(image*255).astype('uint8')
    return image

def visualize(real_image, fake_image):
    real_image=image_to_tensor(real_image)
    fake_image=image_to_tensor(fake_image)
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].imshow(real_image)
    axs[0].set_title('Real Image')
    axs[0].axis('off')
    axs[1].imshow(fake_image)
    axs[1].set_title('Fake Image')
    axs[1].axis('off')
    plt.tight_layout()

    return fig
   

    