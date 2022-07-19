<p align="center">
 <img width=700px src="media/teaser.jpeg" alt="Project logo">
</p>
<h2 align="center">Interactive Surface Flow in Taichi</h2>

 Welcome to my interactive toy of surface flow implemented in [Taichi Lang](https://github.com/taichi-dev/taichi). It creates a real-time fluid simulation on the surface of a triangle mesh manifold with genus 0 based on a slightly modified version of Elcott's [Simplicial Fluids](http://www.geometry.caltech.edu/pubs/ETKSD07.pdf). 

Here are some examples of what it does. It solves the Euler equations in real-time on the mesh surface, and use both colorful dyes and particles to visualize the fluid. I also wrote something over [here](https://yhesper.github.io/projects/2_project_simpfluid/) that briefly explained how my implementation differ from Elcott's algorithm. See below for [user guide](#user-guide). 




https://user-images.githubusercontent.com/35747011/177987263-e3ad4353-075d-43eb-9806-005d86a8f7e3.mp4






https://user-images.githubusercontent.com/35747011/177987322-9f8cdafb-2518-4e9f-9368-18332f417b48.mp4






---

### User Guide

Important: This program is based on [Taichi](https://docs.taichi.graphics/docs/) and its [GGUI](https://docs.taichi-lang.org/docs/ggui). Please make sure your machine satisfies their requirements. Also make sure to install Taichi first!

Warning: the submodule meshtaichi_patcher that this project depends on can only run on mac and linux machines. Sorry windows users. :-( The author of meshtaichi_patcher will be working hard to support windows users, stay tuned!

#### How to Obtain and Build

Clone with:

```
git clone --recursive https://github.com/yhesper/TaichiSimplicialFluid
cd TaichiSimplicialFluid
cd meshtaichi_patcher
pip install -r requirements.txt
python3 setup.py develop --user
cd ..
```


#### How to Play

Run with ```python3 fluid.py meshes/bunny.obj``` (or other obj files in ```meshes``` folder.

* Hold your left mouse button to splat dye and splat force on to the system. 
* Press ```r``` to reset the simulation.
* Press ```p``` to pause.
* Press ```esc``` to exit.
* Press ```x``` to take a snap shot.
* Toggle ```c``` to disable or enable changing dye's color.
* Toggle ```v``` to disable or enable advecting particles to visualize the flow.
* Use ```w,a,s,d``` to move camera, and use your right mouse button to rotate camera. _Right now you can only splat dye and force under the default camera setting, but this would be fixed when the next version of Taichi (1.0.4) is released._


#### Notes

The fluid is advected using an implicit scheme, which is stable. However, excessively adding forces still could result in a very unstable system. If you see ```"Warining: Does not meet Courant–Friedrichs–Lewy condition!"``` being printed out, it means that you have added too much energy to the system  in too short of a period time.

The mesh quality plays a huge part on the behavior of this program, and you should only run this program on meshes that are manifold with no holes. 
