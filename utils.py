
def print_mask(mask, every=8):
    for row in range(0, mask.shape[0], every):
        astr = ''
        num = 0
        for col in range(0, mask.shape[1], every):
            if mask[row, col] == True:
                astr += '#'
            else:
                astr += '0'
        print(astr)
    
def print_mask_val(mask, every=16):
    for row in range(0, mask.shape[0], every):
        astr = ''
        num = 0
        for col in range(0, mask.shape[1], every):
            if mask[row, col] > 128:
                astr += '#'
            else:
                astr += ' '
        print(astr)
    




