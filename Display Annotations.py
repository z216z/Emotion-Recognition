import matplotlib.pyplot as plt

def displayAnnotation(I, body_bbox, anns_con, anns_cat, personID, Npeople, gender, age):
    # I = original size image
    # body_bbox = body bounding box
    # anns_cat --> list holding categories' for the image.
    # anns_con --> list holding valence, arousal and dominance values.

    # subplot(m,n,...)
    m = 5
    n = 11

    x = abs(int(body_bbox[0]))
    y = abs(int(body_bbox[1]))
    w = abs(int(body_bbox[2]) - int(body_bbox[0]))
    h = abs(int(body_bbox[3]) - int(body_bbox[1]))

    # main image with the body bounding box
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(121)
    ax.imshow(I)
    ax.set_title("Annotated person: {}/{}; {}; {}".format(personID, Npeople, gender, age), fontsize=20, fontweight='bold')
    ax.add_patch(plt.Rectangle((x, y), w, h, edgecolor='r', linewidth=5, fill=False))
    print(anns_cat,len(anns_cat),type(anns_cat))
    # plotting categorical annotations
    if len(anns_cat)>3 and type(anns_cat)!= str:
        print(">3")
        l = int((len(anns_cat) + 1) / 2)
        y = list(reversed([i / (l - 1) for i in range(l)]))
        ind = 0
        z = 1
        while ind < len(anns_cat):
            ax = fig.add_subplot(m, n, 46 + z - 1, frameon=False)
            ax.axis('off')
            ax.text(0.5, y[z-1]-1, anns_cat[ind], fontsize=20, fontweight='bold')
            ind += 1
            if ind < len(anns_cat):
                ax.text(0.5, y[z-1]-1, anns_cat[ind], fontsize=20, fontweight='bold')
                ind += 1
            z += 1
    else:
        if type(anns_cat)!=str:
            print("<=3")
            y = [1, 0.6, 0.2]
            ax = fig.add_subplot(m, n, 46, frameon=False)
            ax.axis('off')
            for i, cat in enumerate(list(anns_cat)):
                ax.text(0.5, y[i]-1, cat, fontsize=20, fontweight='bold')
        else:
            y = 1
            ax = fig.add_subplot(m, n, 46, frameon=False)
            ax.axis('off')
            ax.text(0.5, y-1, anns_cat, fontsize=20, fontweight='bold')

    # plotting continuous annotations
    ax = fig.add_subplot(m, n, 51)
    ax.bar(range(3), [anns_con["valence"], anns_con["arousal"], anns_con["dominance"]])
    ax.plot(ax.get_xlim(), [5, 5], '--k')
    ax.set_ylim(0, 10)
    ax.set_xticks(range(3))
    ax.set_xticklabels(['V', 'A', 'D'], fontsize=20, fontweight='bold')

def displayAnnotation_multiple(I, persons):
    for person in persons:
        # font size for the display
        sizeCats = 16
        
        # subplot(m,n,...)
        n = max(len(person["annotations_categories"]),len(person["annotations_continuous"])) + 1
        m = 2
        
        # extract bounding box coordinates
        x = abs(int(person["body_bbox"][0]))
        y = abs(int(person["body_bbox"][1]))
        w = abs(int(person["body_bbox"][2]) - int(person["body_bbox"][0]))
        h = abs(int(person["body_bbox"][3]) - int(person["body_bbox"][1]))
        
        # display image with the body bounding box
        fig = plt.figure(figsize=(20,10))
        ax = fig.add_subplot(m,n,1)
        ax.imshow(I)
        print(person["age"])
        ax.add_patch(plt.Rectangle((x,y),w,h,linewidth=5,edgecolor='r',facecolor='none'))
        plt.title(str(person["age"])+","+person["gender"], fontsize=15)
        
        y = [0.9 - i*0.08 for i in range(len(person["annotations_categories"]["categories"])-1,-1,-1)]
        
        # print discrete annotation per each annotator
        ndisc = len(person["annotations_categories"])
        for a in range(min(len(person["annotations_categories"]),6)):
            ax = fig.add_subplot(m,n,a+2)
            ax.axis('off')
            plt.title(f"Annotator: {a+1}", fontsize=20, fontweight='bold')
            
            # plotting categorical annotations
            for c in range(len(person["annotations_categories"]["categories"])):
                ax.text(0.1, y[c], person["annotations_categories"]["categories"][c], fontsize=sizeCats, fontweight='bold')
                plt.axis('off')
                plt.draw()
        
        # print continuous annotation per each annotator
        for a in range(min(len(person["annotations_continuous"]),6)):
            ax = fig.add_subplot(m,n,n+1+a)
            ax.axis('off')
            plt.title(f"Annotator: {ndisc+a+1}", fontsize=20, fontweight='bold')
            ax.text(0.2, y[0], "Valence:"+str(person["annotations_continuous"]["valence"] / 10), fontsize=sizeCats, fontweight='bold')
            ax.text(0.2, y[1], "Arousal:"+str(person["annotations_continuous"]["arousal"] / 10), fontsize=sizeCats, fontweight='bold')
            ax.text(0.2, y[2], "Dominance:"+str(person["annotations_continuous"]["dominance"] / 10), fontsize=sizeCats, fontweight='bold')
            plt.axis('off')

