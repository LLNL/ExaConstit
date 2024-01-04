#include "mfem.hpp"

#include <cmath>
#include <limits>
#include <type_traits>
#include <algorithm>

using namespace std;
using namespace mfem;

// set the element grain ids from vector data populated from a
// grain map input text file
void setElementGrainIDs(Mesh *mesh, const Vector grainMap, int ncols, int offset);

// used to reset boundary conditions from MFEM convention using
// Make3D() called from the mesh constructor to ExaConstit convention
void setBdrConditions(Mesh *mesh);

// Used to assign BoundaryElement attributes to a rectangular prism in the ExaConstit convention
// z_min = 1, x_min = 2, y_min = 3, z_max = 4, x_max = 5, y_max = 6
void vtkFixBdrElements(Mesh *mesh);

// Borrowed this snippet of code from https://en.cppreference.com/w/cpp/types/numeric_limits/epsilon 
template<class T>
typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type
    almost_equal(T x, T y, int ulp)
{
    // the machine epsilon has to be scaled to the magnitude of the values used
    // and multiplied by the desired precision in ULPs (units in the last place)
    return std::fabs(x-y) <= std::numeric_limits<T>::epsilon() * std::fabs(x+y) * ulp
        // unless the result is subnormal
        || std::fabs(x-y) < std::numeric_limits<T>::min();
}

int main(int argc, char *argv[])
{


   printf("MFEM Version: %d \n", GetVersion());

   // All of our options can be parsed in this file by default
   const char *grain_file = "";
   const char *vtk_file = "";
   const char *output_file = "exaconstit.mesh";
   bool auto_mesh = false;
   // bool vtk_mesh = false;
   int nx, ny, nz;
   nx = ny = nz = 1;
   double lenx, leny, lenz;
   lenx = leny = lenz = 1.0;
   int order;

   Mesh *mesh = nullptr;
   OptionsParser args(argc, argv);
   // All the arguments to automatically generate a mesh
   args.AddOption(&auto_mesh, "-auto_mesh", "--automatic-mesh-generator" ,
                              "-no-auto_mesh", "--no-automatic-mesh-generator", "Enable automatic mesh generation");
   args.AddOption(&nx, "-nx", "--num-elems-xdir", "Number of elements in the x direction");
   args.AddOption(&ny, "-ny", "--num-elems-ydir", "Number of elements in the y direction");
   args.AddOption(&nz, "-nz", "--num-elems-zdir", "Number of elements in the z direction");
   args.AddOption(&lenx, "-lx", "--length-xdir", "Length in the x direction");
   args.AddOption(&leny, "-ly", "--length-ydir", "Length in the y direction");
   args.AddOption(&lenz, "-lz", "--length-zdir", "Length in the z direction");
   // Ability to assign in an element attribute \ grain ID file
   args.AddOption(&grain_file, "-grain", "--grainID-file", "Grain ID or Element Attribute file to use");
   // Ability to read in a VTK file and then assign the appropriate boundary 
   args.AddOption(&vtk_file, "-vtk", "--vtk-file", "VTK v3.0 file that can be read in by MFEM");
   // Output file name
   args.AddOption(&output_file, "-o", "--output-file", "Output file name of the MFEM mesh in the MFEMv1.0 file format");
   // Mesh order
   args.AddOption(&order, "-ord", "--order-elements", "Element order desired for output mesh");

   args.Parse();
   if (!args.Good()) {
      args.PrintUsage(cout);
      return 1;
   }

   if (order < 1) {
      order = 1;
   }

   if(auto_mesh) {
      if (nx <= 0 || ny <= 0 || nz <= 0) {
         MFEM_ABORT("Not all inputted number of elements in each direction was > 0");
      }
      if(abs(lenx) <= 0.0 || abs(leny) <= 0.0 || abs(lenz) <= 0.0) {
         MFEM_ABORT("Not all inputted lengths had a value greater than 0");
      }
      if(grain_file == nullptr){
         MFEM_ABORT("Grain ID or element attribute file was not provided");
      }

      Vector g_map;

      *mesh = mfem::Mesh::MakeCartesian3D(nx, ny, nz, Element::HEXAHEDRON, lenx, leny, lenz, false);

      ifstream igmap(grain_file);
      if (!igmap) {
         cerr << "\nCannot open grain map file: " << grain_file << '\n' << endl;
      }

      int gmapSize = mesh->GetNE();
      g_map.Load(igmap, gmapSize);
      igmap.close();

      // reset boundary conditions from
      setBdrConditions(mesh);

      // set grain ids as element attributes on the mesh
      // The offset of where the grain index is located is
      // location - 1.
      setElementGrainIDs(mesh, g_map, 1, 0);

      mesh->SetCurvature(order);

      ofstream omesh(output_file);
      omesh.precision(14);
      mesh->Print(omesh);

   }

   if (!std::string(vtk_file).empty()) {
      ifstream imesh(vtk_file);
      if (!imesh) {
         cerr << "\nCan not open mesh file: " << vtk_file << endl;
         return 2;
      }
      mesh = new Mesh(imesh, 1, 1, true);
      imesh.close();
      
      const FiniteElementSpace *nodal_fes = mesh->GetNodalFESpace();

      if (nodal_fes != NULL) {
         if(order > nodal_fes->GetOrder(0)) {
            printf("Increasing order of the FE Space to %d\n", order);
            mesh->SetCurvature(order);
         }
      }

      vtkFixBdrElements(mesh);
      ofstream omesh(output_file);
      omesh.precision(14);
      mesh->Print(omesh);
   }

   delete mesh;

   return 0;

}

// set the element grain ids from vector data populated from a
// grain map input text file
void setElementGrainIDs(Mesh *mesh, const Vector grainMap, int ncols, int offset)
{
   // after a call to reorderMeshElements, the elements in the serial
   // MFEM mesh should be ordered the same as the input grainMap
   // vector. Set the element attribute to the grain id. This vector
   // has stride of 4 with the id in the 3rd position indexing from 0

   const double* data = grainMap.HostRead();

   // loop over elements
   for (int i = 0; i<mesh->GetNE(); ++i) {
      mesh->SetAttribute(i, data[ncols * i + offset]);
   }

   return;
}

// used to reset boundary conditions from MFEM convention using
// Make3D() called from the mesh constructor to ExaConstit convention
void setBdrConditions(Mesh *mesh)
{
   // modify MFEM auto cuboidal hex mesh generation boundary
   // attributes to correspond to correct ExaConstit boundary conditions.
   // Look at ../../mesh/mesh.cpp Make3D() to see how boundary attributes
   // are set and modify according to ExaConstit convention

   // loop over boundary elements
   for (int i = 0; i<mesh->GetNBE(); ++i) {
      int bdrAttr = mesh->GetBdrAttribute(i);

      switch (bdrAttr) {
         // note, srw wrote SetBdrAttribute() in ../../mesh/mesh.hpp
         case 1:
            mesh->SetBdrAttribute(i, 1); // bottom
            break;
         case 2:
            mesh->SetBdrAttribute(i, 3); // front
            break;
         case 3:
            mesh->SetBdrAttribute(i, 5); // right
            break;
         case 4:
            mesh->SetBdrAttribute(i, 6); // back
            break;
         case 5:
            mesh->SetBdrAttribute(i, 2); // left
            break;
         case 6:
            mesh->SetBdrAttribute(i, 4); // top
            break;
      }
   }

   return;
}

void vtkFixBdrElements(Mesh *mesh) {
      // modify MFEM auto cuboidal hex mesh generation boundary
   // attributes to correspond to correct ExaConstit boundary conditions.
   // Look at ../../mesh/mesh.cpp Make3D() to see how boundary attributes
   // are set and modify according to ExaConstit convention

   Vector min_len;
   Vector max_len;
   //We don't need to refine this anymore. We're dealing with rectangular prism after all
   mesh->GetBoundingBox(min_len, max_len, 1);

   // loop over boundary elements
   for (int i = 0; i < mesh->GetNBE(); ++i) {
      DenseMatrix bdr_pts;
      mesh->GetBdrPointMatrix(i, bdr_pts);
      for (int vi = 0; vi < 3; vi++) {
         bool bdr_flag = true;
         //First check all of the min vertice points
         for (int vj = 0; vj < bdr_pts.Width(); vj++){
            //break out if a pt doesn't belong on the boundary
            if(!almost_equal(bdr_pts(vi, vj), min_len(vi), 2)) {
               bdr_flag = false;
               break;
            }
         }

         if(bdr_flag) {
            switch (vi) {
               case 0:
                  mesh->SetBdrAttribute(i, 2);
                  break;
               case 1:
                  mesh->SetBdrAttribute(i, 3);
                  break;
               case 2:
                  mesh->SetBdrAttribute(i, 1);
                  break;
            }
            //No need to check the rest of the cases break early
            break;
         }

         bdr_flag = true;
         //Now we check all of the max vertices 
         for (int vj = 0; vj < bdr_pts.Width(); vj++){
            // break out early if the vertices aren't equal
            if(!almost_equal(bdr_pts(vi, vj), max_len(vi), 2)) {
               bdr_flag = false;
               break;
            }
         }
         if(bdr_flag) {
            switch (vi) {
               case 0:
                  mesh->SetBdrAttribute(i, 5);
                  break;
               case 1:
                  mesh->SetBdrAttribute(i, 6);
                  break;
               case 2:
                  mesh->SetBdrAttribute(i, 4);
                  break;
            }
            // No need to check the rest of the cases break early
            break;
         }
      }// end of space_dim loop
   }// end of bdr_elem loop
}// end of vtkFixBdrELements
