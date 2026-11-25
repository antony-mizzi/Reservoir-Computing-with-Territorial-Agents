import numpy as np 
from scipy.spatial import Voronoi
import random as rand 
import matplotlib.pyplot as plt
import maps_and_systems as mapnsys
import Terps_2D as terpo
import time
import scienceplots
plt.style.use(['science', 'no-latex'])
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset


def create_adjacency(homes,mini=-2.5,maxi=2.5):
    vor = Voronoi(homes)
    n = len(homes)
    A = np.zeros((n, n), dtype=bool)

    for ridge, verts in zip(vor.ridge_points, vor.ridge_vertices):
        #print(ridge)
        p1,p2 = vor.vertices[verts]; valid_border = False
        
        #print(verts)
        #print(p1)
        #print(p2)
        if verts[0] != -1:
            if np.min(p1) > mini and np.max(p1) < maxi:
               valid_border = True
        if verts[1] != -1:
            if np.min(p2) > mini and np.max(p2) < maxi:
                valid_border = True
        #print('p1:' +str(p1))
        #print('p2:' +str(p2))
        #print(valid_border)
        #print('')
        if valid_border:
            
            A[ridge[0],ridge[1]] = True
            A[ridge[1],ridge[0]] = True
    return np.where(A, 1, 0)

def graph_territories(vor,homes,TSemb,lag=5):
    adj = create_adjacency(homes)
    #print(adj)
    for (p, q), rv in zip(vor.ridge_points, vor.ridge_vertices):
        #print((p,q))
        #print(rv)
        #print('')
        if -1 not in rv:  # skip infinite ridges
            v0, v1 = vor.vertices[rv]
            if adj[p,q] == 1:
                plt.plot([v0[0], v1[0]], [v0[1], v1[1]], 'k--',linewidth=1)
            else:
                plt.plot([v0[0], v1[0]], [v0[1], v1[1]], 'r--',linewidth=1)
        center = vor.points.mean(axis=0)
    for (p, q), rv in zip(vor.ridge_points, vor.ridge_vertices):
        if -1 in rv:
            i = rv[0] if rv[0] != -1 else rv[1]
            v = vor.vertices[i]

            # Direction perpendicular to the line between the two points
            tangent = vor.points[q] - vor.points[p]
            normal = np.array([-tangent[1], tangent[0]])
            normal /= np.linalg.norm(normal)

            # Choose direction pointing outward
            direction = np.sign(np.dot(v - center, normal)) * normal

            # Extend the infinite edge
            far_point = v + direction * 100
            plt.plot([v[0], far_point[0]], [v[1], far_point[1]], 'k--',linewidth=1)
    
    counter = 0
    for h in homes:
        if counter == 0:
            plt.plot(h[0],h[1],'x',color='C0')
        elif adj[0,counter] == 1:
            #plt.text(h[0],h[1],str(counter),color='green')
            plt.plot(h[0],h[1],'x',color='green')
        else:
            plt.plot(h[0],h[1],'x',color='k')
            #plt.text(h[0],h[1],str(counter),color='k')
        counter += 1
    for point in vor.vertices:
        plt.plot([point[0]],[point[1]],'o',color='r',markersize=3)
    plt.plot([-2.5,-2.5],[-2.5,2.5],color='C0')
    plt.plot([2.5,2.5],[-2.5,2.5],color='C0')
    plt.plot([-2.5,2.5],[-2.5,-2.5],color='C0')
    plt.plot([-2.5,2.5],[2.5,2.5],color='C0')
    #borders = get_all_borders(vor)
    #for border in borders:
    #    plt.plot([border[0,0]],[border[0,1]],'*',color='C0')
    #    plt.plot([border[1,0]],[border[1,1]],'*',color='C1')
    #    plt.plot([border[0,0], border[1,0]], [border[0,1], border[1,1]], 'r--',linewidth=1,alpha=0.3)
    plt.xlim([-4,4])
    plt.ylim([-4,4])
    plt.show()
    
def graph_territorial_border_change(vor1,homes1,vor2,homes2,crossings,TSemb,lag=5):
    fig,axs=plt.subplots(1,2,figsize=(16,8))
    TS = TSemb[:,0]
    axs[1].plot(TS[:-lag],TS[lag:],color='k',alpha=0.2)
    
    
    crossings = 0.8*crossings/np.max(crossings) + 0.19*np.ones((len(homes1),len(homes1)))
    axs[0].plot(TS[:-lag],TS[lag:],color='k',alpha=0.2)
    center = vor1.points.mean(axis=0)
    for (p, q), rv in zip(vor1.ridge_points, vor1.ridge_vertices):
        if -1 not in rv:  # skip infinite ridges
            v0, v1 = vor1.vertices[rv]
            axs[0].plot([v0[0], v1[0]], [v0[1], v1[1]], 'k--',linewidth=1,alpha=1)#crossings[p,q])
            if crossings[p,q] == np.max(crossings):
                axs[0].plot([v0[0], v1[0]], [v0[1], v1[1]], 'r--',linewidth=2,alpha=1)
        else:
            i = rv[0] if rv[0] != -1 else rv[1]
            v = vor1.vertices[i]

            # Direction perpendicular to the line between the two points
            tangent = vor1.points[q] - vor1.points[p]
            normal = np.array([-tangent[1], tangent[0]])
            normal /= np.linalg.norm(normal)

            # Choose direction pointing outward
            direction = np.sign(np.dot(v - center, normal)) * normal

            # Extend the infinite edge
            far_point = v + direction * 100
            axs[0].plot([v[0], far_point[0]], [v[1], far_point[1]], 'k--',linewidth=1,alpha=1)#crossings[p,q])
            if crossings[p,q] == np.max(crossings):
                axs.plot([v0[0], v1[0]], [v0[1], v1[1]], 'r--',linewidth=2,alpha=1)
    
    counter = 0
    for h in homes1:
        #plt.text(h[0],h[1],str(counter),color='k')
        axs[0].plot(h[0],h[1],'x',color='C0')
        counter += 1
    axs[0].set_xlim([-3,3])
    axs[0].set_ylim([-3,3])
    axs[0].set_yticks([])
    axs[0].set_xticks([])
    
    center = vor2.points.mean(axis=0)
    for (p, q), rv in zip(vor2.ridge_points, vor2.ridge_vertices):
        if -1 not in rv:  # skip infinite ridges
            v0, v1 = vor2.vertices[rv]
            axs[1].plot([v0[0], v1[0]], [v0[1], v1[1]], 'k--',linewidth=1,alpha=1)
        else:
            i = rv[0] if rv[0] != -1 else rv[1]
            v = vor2.vertices[i]

            # Direction perpendicular to the line between the two points
            tangent = vor2.points[q] - vor2.points[p]
            normal = np.array([-tangent[1], tangent[0]])
            normal /= np.linalg.norm(normal)

            # Choose direction pointing outward
            direction = np.sign(np.dot(v - center, normal)) * normal

            # Extend the infinite edge
            far_point = v + direction * 100
            axs[1].plot([v[0], far_point[0]], [v[1], far_point[1]], 'k--',linewidth=1,alpha=1)
            
    counter = 0
    for h in homes2:
        #plt.text(h[0],h[1],str(counter),color='k')
        axs[1].plot(h[0],h[1],'x',color='C0')
        counter += 1
    axs[1].set_xlim([-3,3])
    axs[1].set_ylim([-3,3])
    axs[1].set_yticks([])
    axs[1].set_xticks([])
    plt.tight_layout()
    plt.show()
    
def graph_most_crossed_borders(vor,homes,TSemb,crossings,lag=5):
    fig,axs=plt.subplots(1,1,figsize=(8,8))
    TS = TSemb[:,0]
    crossings = 0.8*crossings/np.max(crossings) + 0.19*np.ones((len(homes),len(homes)))
    axs.plot(TS[:-lag],TS[lag:],color='k',alpha=0.2)
    center = vor.points.mean(axis=0)
    for (p, q), rv in zip(vor.ridge_points, vor.ridge_vertices):
        if -1 not in rv:  # skip infinite ridges
            v0, v1 = vor.vertices[rv]
            axs.plot([v0[0], v1[0]], [v0[1], v1[1]], 'k--',linewidth=1,alpha=1)#crossings[p,q])
            if crossings[p,q] == np.max(crossings):
                axs.plot([v0[0], v1[0]], [v0[1], v1[1]], 'r--',linewidth=2,alpha=1)
        else:
            i = rv[0] if rv[0] != -1 else rv[1]
            v = vor.vertices[i]

            # Direction perpendicular to the line between the two points
            tangent = vor.points[q] - vor.points[p]
            normal = np.array([-tangent[1], tangent[0]])
            normal /= np.linalg.norm(normal)

            # Choose direction pointing outward
            direction = np.sign(np.dot(v - center, normal)) * normal

            # Extend the infinite edge
            far_point = v + direction * 100
            axs.plot([v[0], far_point[0]], [v[1], far_point[1]], 'k--',linewidth=1,alpha=1)#crossings[p,q])
            if crossings[p,q] == np.max(crossings):
                axs.plot([v0[0], v1[0]], [v0[1], v1[1]], 'r--',linewidth=2,alpha=1)
    
    counter = 0
    for h in homes:
        #plt.text(h[0],h[1],str(counter),color='k')
        plt.plot(h[0],h[1],'x',color='C0')
        counter += 1
    axs.set_xlim([-3,3])
    axs.set_ylim([-3,3])
    axs.set_yticks([])
    axs.set_xticks([])
    plt.show()
    
def graph_borders(vor,homes,TSemb,lag=5):
    fig,axs=plt.subplots(1,1,figsize=(8,8))
    TS = TSemb[:,0]

    axs.plot(TS[:-lag],TS[lag:],color='k',alpha=0.2)
    center = vor.points.mean(axis=0)
    for (p, q), rv in zip(vor.ridge_points, vor.ridge_vertices):
        if -1 not in rv:  # skip infinite ridges
            v0, v1 = vor.vertices[rv]
            axs.plot([v0[0], v1[0]], [v0[1], v1[1]], 'k--',linewidth=1,alpha=1)#crossings[p,q])
        
        else:
            i = rv[0] if rv[0] != -1 else rv[1]
            v = vor.vertices[i]

            # Direction perpendicular to the line between the two points
            tangent = vor.points[q] - vor.points[p]
            normal = np.array([-tangent[1], tangent[0]])
            normal /= np.linalg.norm(normal)

            # Choose direction pointing outward
            direction = np.sign(np.dot(v - center, normal)) * normal

            # Extend the infinite edge
            far_point = v + direction * 100
            axs.plot([v[0], far_point[0]], [v[1], far_point[1]], 'k--',linewidth=1,alpha=1)#crossings[p,q])
    
    counter = 0
    for h in homes:
        #plt.text(h[0],h[1],str(counter),color='k')
        plt.plot(h[0],h[1],'x',color='C0')
        counter += 1
    axs.set_xlim([-3,3])
    axs.set_ylim([-3,3])
    axs.set_yticks([])
    axs.set_xticks([])
    plt.show()
    
def get_all_borders(vor):
    borders = []; center = vor.points.mean(axis=0)
    for (p, q), rv in zip(vor.ridge_points, vor.ridge_vertices):
        if -1 not in rv:  # skip infinite ridges
            v0, v1 = vor.vertices[rv]
            borders.append(vor.vertices[rv])
        else:
            i = rv[0] if rv[0] != -1 else rv[1]
            v = vor.vertices[i]

            # Direction perpendicular to the line between the two points
            tangent = vor.points[q] - vor.points[p]
            normal = np.array([-tangent[1], tangent[0]])
            normal /= np.linalg.norm(normal)

            # Choose direction pointing outward
            direction = np.sign(np.dot(v - center, normal)) * normal

            # Extend the infinite edge
            far_point = v + direction * 4
            borders.append(np.array([[v[0],v[1]], [far_point[0], far_point[1]]]))
    return borders
    

def method_of_parallel_flows(TSemb,homes,lag=5,scale=0.2,plotstuff=False):
    vor = Voronoi(homes)
    #graph_territories(vor, homes, TSemb)
    #borders = get_all_borders(vor)
    N = len(homes)
    border_crossings = np.zeros((N,N))
    false_crossings = np.zeros((N,N))
    TS = TSemb[:,0]
    
    for i in range(TSemb.shape[0]-1):
        v = TSemb[i+1] - TSemb[i]
        oldcell = np.argmin(np.linalg.norm(homes-TSemb[i],axis=1))
        truecell = np.argmin(np.linalg.norm(homes-TSemb[i+1],axis=1))
        if oldcell != truecell:
                border_crossings[truecell,oldcell] += 1
                border_crossings[oldcell,truecell] += 1
        perp = scale*np.array([-v[1],v[0]])
        points = [TSemb[i+1]+perp, TSemb[i+1]-perp]
        cells = [np.argmin(np.linalg.norm(homes-p,axis=1)) for p in points]
        for c in cells:
            if c != truecell:
                
                false_crossings[truecell,c] += 1
                false_crossings[c,truecell] += 1
                
                if plotstuff:
                    plt.plot(TS[:-lag],TS[lag:],color='k',alpha=0.2)
                    plt.plot([TSemb[i][0]],[TSemb[i][1]],'x',color='r',markersize=2)
                    plt.plot([TSemb[i+1][0]],[TSemb[i+1][1]],'o',color='C1',markersize=2)
                    for p in points:
                        plt.plot([p[0]],[p[1]],'o',color='C1',markersize=2)
                    
                    for ridge in vor.ridge_vertices:
                        if -1 not in ridge:  # skip infinite ridges
                            v0, v1 = vor.vertices[ridge]
                            plt.plot([v0[0], v1[0]], [v0[1], v1[1]], 'k--',linewidth=1)
                        center = vor.points.mean(axis=0)
                    for (p, q), rv in zip(vor.ridge_points, vor.ridge_vertices):
                        if -1 in rv:
                            j = rv[0] if rv[0] != -1 else rv[1]
                            v = vor.vertices[j]
    
                            # Direction perpendicular to the line between the two points
                            tangent = vor.points[q] - vor.points[p]
                            normal = np.array([-tangent[1], tangent[0]])
                            normal /= np.linalg.norm(normal)
    
                            # Choose direction pointing outward
                            direction = np.sign(np.dot(v - center, normal)) * normal
    
                            # Extend the infinite edge
                            far_point = v + direction * 100
                            plt.plot([v[0], far_point[0]], [v[1], far_point[1]], 'k--',linewidth=1)
                    plt.xlim([-3,3])
                    plt.ylim([-3,3])
                    plt.title('Timestep '+str(i))
                    plt.show()
                #time.sleep(1)
    border_crossings += np.ones((N,N))*10**(-5)
    border_crossings += false_crossings 
    return np.multiply(np.reciprocal(border_crossings), false_crossings)


def method_of_near_neighbour_crossings(TSemb,homes,lag=5,scale=0.2,plotstuff=False):
    vor = Voronoi(homes)
    #graph_territories(vor, homes, TSemb)
    #borders = get_all_borders(vor)
    N = len(homes)
    border_crossings = np.zeros((N,N))
    false_crossings = np.zeros((N,N))
    TS = TSemb[:,0]
    
    for i in range(TSemb.shape[0]-1):
        dists = np.sum((TSemb[:-1,:] - TSemb[i])**2, axis=1)
        inds = np.argpartition(dists, 3)[:3]
        steps = [TSemb[j+1]-TSemb[j] for j in inds]

        oldcell = np.argmin(np.linalg.norm(homes-TSemb[i],axis=1))
        points = [TSemb[i] + s for s in steps]
        newcells = list(set([np.argmin(np.linalg.norm(homes-p,axis=1)) for p in points]))
        if len(newcells) == 1:
            if newcells[0] != oldcell:    
                border_crossings[oldcell,newcells[0]] += 1
                border_crossings[newcells[0],oldcell] += 1
        else:
            if len(newcells) == 2:
                false_crossings[newcells[0],newcells[1]] += 1
                false_crossings[newcells[1],newcells[0]] += 1
            elif len(newcells) == 3:
                false_crossings[newcells[0],newcells[1]] += 1
                false_crossings[newcells[1],newcells[0]] += 1
                
                false_crossings[newcells[0],newcells[2]] += 1
                false_crossings[newcells[2],newcells[0]] += 1
                
                false_crossings[newcells[2],newcells[1]] += 1
                false_crossings[newcells[1],newcells[2]] += 1
                
                
            if plotstuff:
                plt.plot(TS[:-lag],TS[lag:],color='k',alpha=0.2)
                for point in points:
                    plt.plot([point[0]],[point[1]],'o',color='C0',markersize=2)
                    plt.plot([TSemb[i][0],point[0]],[TSemb[i][1],point[1]],color='C0',linewidth=2)
                #for ind in inds:
                #    plt.plot([TSemb[ind][0]],[TSemb[ind][1]],'o',color='C0',markersize=2)
                #    plt.plot([TSemb[ind+1][0]],[TSemb[ind+1][1]],'o',color='C0',markersize=2)
                #    plt.plot([TSemb[ind][0],TSemb[ind+1][0]],[TSemb[ind][1],TSemb[ind+1][1]],color='C0',linewidth=2)
                plt.plot([TSemb[i][0]],[TSemb[i][1]],'x',color='r',markersize=2)
                #plt.plot([TSemb[i+1][0]],[TSemb[i+1][1]],'o',color='C1',markersize=2)
                
                
                
                for ridge in vor.ridge_vertices:
                    if -1 not in ridge:  # skip infinite ridges
                        v0, v1 = vor.vertices[ridge]
                        plt.plot([v0[0], v1[0]], [v0[1], v1[1]], 'k--',linewidth=1)
                    center = vor.points.mean(axis=0)
                for (p, q), rv in zip(vor.ridge_points, vor.ridge_vertices):
                    if -1 in rv:
                        j = rv[0] if rv[0] != -1 else rv[1]
                        v = vor.vertices[j]

                        # Direction perpendicular to the line between the two points
                        tangent = vor.points[q] - vor.points[p]
                        normal = np.array([-tangent[1], tangent[0]])
                        normal /= np.linalg.norm(normal)

                        # Choose direction pointing outward
                        direction = np.sign(np.dot(v - center, normal)) * normal

                        # Extend the infinite edge
                        far_point = v + direction * 100
                        plt.plot([v[0], far_point[0]], [v[1], far_point[1]], 'k--',linewidth=1)
                counter = 0
                for h in homes:
                    plt.text(h[0],h[1],str(counter),color='k')
                    counter += 1
                plt.xlim([-3,3])
                plt.ylim([-3,3])
                plt.title('Timestep '+str(i))
                plt.show()
                #time.sleep(5)
    border_crossings += np.ones((N,N))*10**(1)
    #border_crossings += false_crossings 
    return np.multiply(np.reciprocal(border_crossings), false_crossings)


def method_of_first_false_crossings(TSemb,homes,lag=5,scale=0.2,plotstuff=False, repeat = False,
                                    new_homes = None, final_crossings = None, only_most_crossed = False):
    
    vor = Voronoi(homes)
    if repeat:
        vor2 = Voronoi(new_homes)
        N2 = len(new_homes)
    N = len(homes)
    correct_crossings = np.zeros((N,N))
    false_crossings = np.zeros((N,N))
    TS = TSemb[:,0]
    mask_maker = np.array(list(range(len(TS))))
    
    for i in range(0,TSemb.shape[0]-20,2):
        if rand.random() > 0.5:
            mask = (abs(mask_maker-i) > 11) & (mask_maker < len(TS) - 11)
            #print('length of mask is '+str(len(mask)))
            
            dists = np.sum((TSemb[:-1,:] - TSemb[i])**2, axis=1)
            dists = np.where(mask[:-1],dists,100)
            #print(dists)
            #time.sleep(1)
            
            t_prime = np.argmin(dists); delta = 1; keep_going = True
            dvec = TSemb[i]-TSemb[t_prime]
            recent_cell = np.argmin(np.linalg.norm(homes-TSemb[i],axis=1))
            found_crossing = False; crossed_border = [100,100]
            if np.argmin(np.linalg.norm(homes-TSemb[t_prime],axis=1)) != recent_cell:
                keep_going = False
            
            while keep_going:
                traj1 = TSemb[i+delta]; traj2 = TSemb[t_prime+delta]+dvec
                cell1 = np.argmin(np.linalg.norm(homes-traj1,axis=1))
                cell2 = np.argmin(np.linalg.norm(homes-traj2,axis=1))
                if cell1 == cell2:
                    if cell1 != recent_cell:
                        correct_crossings[recent_cell,cell1] += 1
                        correct_crossings[cell1,recent_cell] += 1
                        recent_cell = cell1+0
                else:
                    false_crossings[cell1,cell2] += 1
                    false_crossings[cell2,cell1] += 1
                    found_crossing = True
                    crossed_border = [cell1,cell2]
                    keep_going = False
                delta += 1
                if delta > 10:
                    keep_going = False
                
                    
                    
            if plotstuff and found_crossing:# and rand.random() < 0.1:
                showplot = True 
                if only_most_crossed:
                    showplot = False
                    if N*crossed_border[0]+crossed_border[1] == np.argmax(final_crossings):
                        showplot = True
                    elif N*crossed_border[1]+crossed_border[0] == np.argmax(final_crossings):
                        showplot = True
                    else:
                        print(str(crossed_border)+' ('+str(N*crossed_border[0]+crossed_border[1])+
                              ','+str(N*crossed_border[1]+crossed_border[0])+')'+' '+
                              str(np.argmax(final_crossings)))
                if showplot:
                    
                    fig,axs=plt.subplots(1,1,figsize=(8,8),dpi=120)
                    axins = zoomed_inset_axes(axs, zoom=4.5, loc='lower right')
                    x = TSemb[t_prime:t_prime+delta,0] + dvec[0]
                    y = TSemb[t_prime:t_prime+delta,1] + dvec[1]
                    #axins=axs[counter//2,counter%2].inset_axes([np.exp(-1),np.exp(-1),np.exp(1),np.exp(1)])
                    xint =  TSemb[i+delta-1,0]; yint = TSemb[i+delta-1,1]
                    #axins.plot(Surescores,just_avg(mizVs,log=False),'x',color='k',markersize=12,alpha=1)
                   
                    axins.set_xticks([])
                    axins.set_yticks([])
                    axins.set_xlim(xint-0.2,xint+0.2)
                    axins.set_ylim(yint-0.2,yint+0.2)
                    axs.plot(TS[:-lag],TS[lag:],color='k',alpha=0.2)
                    axins.plot(TS[:-lag],TS[lag:],color='k',alpha=0.2)
                    
                    axs.plot(TSemb[i:i+delta,0],TSemb[i:i+delta,1],color='k',linewidth=1,label='$\\vec{u}(t)$')
                    #axs.plot(TSemb[t_prime:t_prime+delta,0],TSemb[t_prime:t_prime+delta,1],color='C0',linewidth=1,
                    #         label="$\\vec{u}(t')$")
                    axins.plot(TSemb[i:i+delta,0],TSemb[i:i+delta,1],color='k',linewidth=1)
                    #axins.plot(TSemb[t_prime:t_prime+delta,0],TSemb[t_prime:t_prime+delta,1],color='C0',linewidth=1)
        
                    axs.plot(x,y,color='C0',linewidth=1,label="$\\vec{u}(t')+\\vec{d}$")
                    axs.legend(fontsize=18)
                    axins.plot(x,y,color='C0',linewidth=1)
                    if found_crossing:
                        axs.plot([x[delta-1]],[y[delta-1]],'o',color='C0')
                        axs.plot([TSemb[i+delta-1,0]],[TSemb[i+delta-1,1]],'o',color='k')
                        axins.plot([x[delta-1]],[y[delta-1]],'o',color='C0')
                        axins.plot([TSemb[i+delta-1,0]],[TSemb[i+delta-1,1]],'o',color='k')
                        
                    
                    for (p,q), ridge in zip(vor.ridge_points, vor.ridge_vertices):
                        if -1 not in ridge:  # skip infinite ridges
                            v0, v1 = vor.vertices[ridge]
                            if p in crossed_border and q in crossed_border:
                                linetype = 'r--'
                            else:
                                linetype = 'k--'
                            axs.plot([v0[0], v1[0]], [v0[1], v1[1]], linetype,linewidth=1)
                            axins.plot([v0[0], v1[0]], [v0[1], v1[1]], linetype,linewidth=1)
                        center = vor.points.mean(axis=0)
                    for (p, q), rv in zip(vor.ridge_points, vor.ridge_vertices):
                        if -1 in rv:
                            j = rv[0] if rv[0] != -1 else rv[1]
                            v = vor.vertices[j]
        
                            # Direction perpendicular to the line between the two points
                            tangent = vor.points[q] - vor.points[p]
                            normal = np.array([-tangent[1], tangent[0]])
                            normal /= np.linalg.norm(normal)
        
                            # Choose direction pointing outward
                            direction = np.sign(np.dot(v - center, normal)) * normal
        
                            # Extend the infinite edge
                            far_point = v + direction * 100
                            axs.plot([v[0], far_point[0]], [v[1], far_point[1]], 'k--',linewidth=1)
                            axins.plot([v[0], far_point[0]], [v[1], far_point[1]], 'k--',linewidth=1)
                    #counter = 0
                    #for h in homes:
                    #    plt.text(h[0],h[1],str(counter),color='k')
                    #    counter += 1
                    
                    axs.set_xlim([-3,3])
                    axs.set_ylim([-3,3])
                    axs.set_yticks([])
                    axs.set_xticks([])
                    
                    for h in homes:
                        axs.plot(h[0],h[1],'x',color='C1')
                        
                    
                    mark_inset(axs, axins, loc1=2, loc2=4, fc="none", ec="0.5")
                    #plt.title('Timestep '+str(i))
                    plt.show()
                    
                    if repeat:
                        fig,axs=plt.subplots(1,1,figsize=(8,8),dpi=120)
                        axins = zoomed_inset_axes(axs, zoom=4.5, loc='lower right')
                        x = TSemb[t_prime:t_prime+delta,0] + dvec[0]
                        y = TSemb[t_prime:t_prime+delta,1] + dvec[1]
                        #axins=axs[counter//2,counter%2].inset_axes([np.exp(-1),np.exp(-1),np.exp(1),np.exp(1)])
                        xint =  TSemb[i+delta-1,0]; yint = TSemb[i+delta-1,1]
                        #axins.plot(Surescores,just_avg(mizVs,log=False),'x',color='k',markersize=12,alpha=1)
                       
                        axins.set_xticks([])
                        axins.set_yticks([])
                        axins.set_xlim(xint-0.2,xint+0.2)
                        axins.set_ylim(yint-0.2,yint+0.2)
                        axs.plot(TS[:-lag],TS[lag:],color='k',alpha=0.2)
                        axins.plot(TS[:-lag],TS[lag:],color='k',alpha=0.2)
                        
                        axs.plot(TSemb[i:i+delta,0],TSemb[i:i+delta,1],color='k',linewidth=1,label='$\\vec{u}(t)$')
                        #axs.plot(TSemb[t_prime:t_prime+delta,0],TSemb[t_prime:t_prime+delta,1],color='C0',linewidth=1,
                        #         label="$\\vec{u}(t')$")
                        axins.plot(TSemb[i:i+delta,0],TSemb[i:i+delta,1],color='k',linewidth=1)
                        #axins.plot(TSemb[t_prime:t_prime+delta,0],TSemb[t_prime:t_prime+delta,1],color='C0',linewidth=1)
            
                        axs.plot(x,y,color='C0',linewidth=1,label="$\\vec{u}(t')+\\vec{d}$")
                        axs.legend(fontsize=18)
                        axins.plot(x,y,color='C0',linewidth=1)
                        if found_crossing:
                            axs.plot([x[delta-1]],[y[delta-1]],'o',color='C0')
                            axs.plot([TSemb[i+delta-1,0]],[TSemb[i+delta-1,1]],'o',color='k')
                            axins.plot([x[delta-1]],[y[delta-1]],'o',color='C0')
                            axins.plot([TSemb[i+delta-1,0]],[TSemb[i+delta-1,1]],'o',color='k')
                            
                        
                        for (p,q), ridge in zip(vor2.ridge_points, vor2.ridge_vertices):
                            if -1 not in ridge:  # skip infinite ridges
                                v0, v1 = vor2.vertices[ridge]; linetype = 'k--'
                                #if p in crossed_border and q in crossed_border:
                                #    linetype = 'r--'
                                #else:
                                #    linetype = 'k--'
                                axs.plot([v0[0], v1[0]], [v0[1], v1[1]], linetype,linewidth=1)
                                axins.plot([v0[0], v1[0]], [v0[1], v1[1]], linetype,linewidth=1)
                            center = vor2.points.mean(axis=0)
                        for (p, q), rv in zip(vor2.ridge_points, vor2.ridge_vertices):
                            if -1 in rv:
                                j = rv[0] if rv[0] != -1 else rv[1]
                                v = vor2.vertices[j]
            
                                # Direction perpendicular to the line between the two points
                                tangent = vor2.points[q] - vor2.points[p]
                                normal = np.array([-tangent[1], tangent[0]])
                                normal /= np.linalg.norm(normal)
            
                                # Choose direction pointing outward
                                direction = np.sign(np.dot(v - center, normal)) * normal
            
                                # Extend the infinite edge
                                far_point = v + direction * 100
                                axs.plot([v[0], far_point[0]], [v[1], far_point[1]], 'k--',linewidth=1)
                                axins.plot([v[0], far_point[0]], [v[1], far_point[1]], 'k--',linewidth=1)
                        #counter = 0
                        #for h in homes:
                        #    plt.text(h[0],h[1],str(counter),color='k')
                        #    counter += 1
                        
                        axs.set_xlim([-3,3])
                        axs.set_ylim([-3,3])
                        axs.set_yticks([])
                        axs.set_xticks([])
                        
                        for h in new_homes:
                            axs.plot(h[0],h[1],'x',color='C1')
                        
                        mark_inset(axs, axins, loc1=2, loc2=4, fc="none", ec="0.5")
                        #plt.title('Timestep '+str(i))
                        plt.show()
                
                #time.sleep(5)
    correct_crossings += np.ones((N,N))*10**(0)
    correct_crossings += false_crossings 
    return np.multiply(np.reciprocal(correct_crossings), false_crossings)


def delete_by_parallel_crossings(TSemb,homes,lag=5,scale=0.2):
    crossings = method_of_parallel_flows(TSemb, homes, scale = scale)
    idx = np.unravel_index(np.argmax(crossings), crossings.shape)
    maxi = max(idx); mini = min(idx)
    midpoint = 0.5*(homes[maxi]+homes[mini])
    homes = np.delete(homes, maxi, axis=0)    
    homes = np.delete(homes, mini, axis=0)  
    homes = np.append(homes, midpoint.reshape(1, 2), axis=0)
    return homes

def delete_by_false_near_neighbour_crossings(TSemb,homes,lag=5,scale=0.2):
    crossings = method_of_near_neighbour_crossings(TSemb,homes,lag=lag)
    idx = np.unravel_index(np.argmax(crossings), crossings.shape)
    maxi = max(idx); mini = min(idx)
    midpoint = 0.5*(homes[maxi]+homes[mini])
    homes = np.delete(homes, maxi, axis=0)    
    homes = np.delete(homes, mini, axis=0)  
    homes = np.append(homes, midpoint.reshape(1, 2), axis=0)
    return homes

def delete_by_first_false_crossings(TSemb,homes,lag=5,scale=0.2):
    crossings = method_of_first_false_crossings(TSemb,homes,lag=lag)
    idx = np.unravel_index(np.argmax(crossings), crossings.shape)
    maxi = max(idx); mini = min(idx)
    midpoint = 0.5*(homes[maxi]+homes[mini])
    homes = np.delete(homes, maxi, axis=0)    
    homes = np.delete(homes, mini, axis=0)  
    homes = np.append(homes, midpoint.reshape(1, 2), axis=0)
    return homes

def merge_random_neighbours(homes):
    idx = rand.randint(0,homes.shape[0]-1)
    vor = Voronoi(homes)
    neighbours = []

    for ridge in vor.ridge_points:
        if idx in ridge:
            neighbours.append(ridge)
    ridge = rand.choice(neighbours)
    maxi = max(ridge); mini = min(ridge)
    midpoint = 0.5*(homes[maxi]+homes[mini])
    homes = np.delete(homes, maxi, axis=0)    
    homes = np.delete(homes, mini, axis=0)  
    homes = np.append(homes, midpoint.reshape(1, 2), axis=0)
    return homes
    
   
def Illustrate_Deletion(TSemb,homes,lag=5,scale=0.2):
    vor = Voronoi(homes)
    crossings = method_of_first_false_crossings(TSemb,homes,lag=lag,plotstuff=False)
    print(crossings)
    print(np.argmax(crossings))
    #graph_most_crossed_borders(vor, homes, TSemb, crossings, lag = lag)
    idx = np.unravel_index(np.argmax(crossings), crossings.shape)
    maxi = max(idx); mini = min(idx)
    midpoint = 0.5*(homes[maxi]+homes[mini])
    homes2 = np.delete(homes, maxi, axis=0)    
    homes2 = np.delete(homes2, mini, axis=0)  
    homes2 = np.append(homes2, midpoint.reshape(1, 2), axis=0)
    vor2 = Voronoi(homes2)
    #graph_territorial_border_change(vor,homes,vor2,homes2,crossings,TSemb,lag=7)
    graph_borders(vor2, homes2, TSemb,lag=lag)
    graph_most_crossed_borders(vor, homes, TSemb, crossings,lag=lag)
    
    time.sleep(5)
    crossings = method_of_first_false_crossings(TSemb,homes,lag=lag,plotstuff=True,repeat=True,
                                                new_homes = homes2, final_crossings=crossings,
                                                only_most_crossed=True)
    return homes

#TS = mapnsys.runge_Lorenz(3000)
#homes = np.random.uniform(-2.5,2.5,size=(20,2))
#vor = Voronoi(homes)
#borders = get_all_borders(vor)
#terps = terpo.Terp_land2d(homes)
#TSemb,Y = terps.embedd_TS(TS,lag = 7)


#Illustrate_Deletion(TSemb, homes, lag=7)
#crossings = method_of_first_false_crossings(TSemb, homes,plotstuff=True, lag=7)
#print(crossings)
#idx = np.unravel_index(np.argmax(crossings), crossings.shape)
#print('IDX IS '+str(idx))
#vor = Voronoi(homes)
#graph_most_crossed_borders(vor, homes, TSemb, crossings, lag = 7)

#crossings = method_of_near_neighbour_crossings(TSemb, homes,plotstuff=False, lag=5)
#print(crossings)
#idx = np.unravel_index(np.argmax(crossings), crossings.shape)
#print('IDX IS '+str(idx))
#vor = Voronoi(homes)
#graph_most_crossed_borders(vor, homes, TSemb, crossings, lag = 5)

#crossings = method_of_parallel_flows(TSemb, homes,plotstuff=False,scale=0.5)
#idx = np.unravel_index(np.argmax(crossings), crossings.shape)
#print(crossings)
#print('IDX IS '+str(idx))
#vor = Voronoi(homes)
#graph_most_crossed_borders(vor, homes, TSemb, crossings, lag = 7)

#homes = delete_by_parallel_crossings(TSemb, homes,scale=0.2)
#vor = Voronoi(homes)
#crossings = method_of_parallel_flows(TSemb, homes,scale=0.2)
#graph_most_crossed_borders(vor, homes, TSemb, crossings)





















