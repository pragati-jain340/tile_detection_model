 # (3, H, W)
            imgB = np.repeat(imgB[None, ...], 3, axis=0)
        else:
            imgA = imgA[None, ...]  # (1, H, W)
            imgB = imgB[None, ...]