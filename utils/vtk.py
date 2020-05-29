import vtk
import numpy as np
from vtk.util import numpy_support


def vtk_to_numpy(image):

    if image.GetNumberOfScalarComponents() == 4:
        dims = image.GetDimensions()
        points_data = image.GetPointData().GetScalars()
        array = numpy_support.vtk_to_numpy(points_data)
        array = array.reshape(dims[2], dims[1], dims[0], 4)
        array = array.transpose(3, 2, 1, 0)
    else:
        dims = image.GetDimensions()
        points_data = image.GetPointData().GetScalars()

        array = numpy_support.vtk_to_numpy(points_data)
        array = array.reshape(dims[2], dims[1], dims[0])
        array = array.transpose(2, 1, 0)
    return array


def numpy_to_vtk(array, spacing=(1., 1., 1.)):

    if array.ndim == 3:
        dim = array.shape
        # Transpose array to ZYX format to get string representation
        array_string = array.transpose(2, 1, 0).astype(np.float32).tostring()
        importer = vtk.vtkImageImport()
        importer.CopyImportVoidPointer(array_string, len(array_string))
        importer.SetDataScalarType(vtk.VTK_FLOAT)
        importer.SetNumberOfScalarComponents(1)
        extent = importer.GetDataExtent()

    else:
        dim = array.shape[1:]
        array_string = array.transpose(3, 2, 1, 0).astype(np.float32).tostring()
        importer = vtk.vtkImageImport()
        importer.CopyImportVoidPointer(array_string, len(array_string))
        importer.SetDataScalarType(vtk.VTK_FLOAT)
        importer.SetNumberOfScalarComponents(4)
        extent = importer.GetDataExtent()

    importer.SetDataExtent(extent[0], extent[0] + dim[0] - 1,
                           extent[2], extent[2] + dim[1] - 1,
                           extent[4], extent[4] + dim[2] - 1)

    importer.SetWholeExtent(extent[0], extent[0] + dim[0] - 1,
                            extent[2], extent[2] + dim[1] - 1,
                            extent[4], extent[4] + dim[2] - 1)

    importer.SetDataSpacing(spacing[0], spacing[1], spacing[2])
    importer.SetDataOrigin(0, 0, 0)

    importer.Update()
    image_data = importer.GetOutput()

    ret = vtk.vtkImageData()
    ret.DeepCopy(image_data)
    return ret


def show_image(image, actors=[]):
    viewer = vtk.vtkImageViewer2()
    viewer.SetInputData(image)

    istyle = vtk.vtkInteractorStyleImage()
    iren = vtk.vtkRenderWindowInteractor()
    viewer.SetupInteractor(iren)
    iren.SetInteractorStyle(istyle)

    min_slice = viewer.GetSliceMin()
    max_slice = viewer.GetSliceMax()

    actions = {'slice': min_slice}

    def mouse_wheel_forward_event(caller, event):
        if actions['slice'] > min_slice:
            actions['slice'] -= 1
            viewer.SetSlice(actions['slice'])
            viewer.Render()

    def mouse_wheel_backward_event(caller, event):
        if actions['slice'] < max_slice:
            actions['slice'] += 1
            viewer.SetSlice(actions['slice'])
            viewer.Render()

    istyle.AddObserver('MouseWheelForwardEvent', mouse_wheel_forward_event)
    istyle.AddObserver('MouseWheelBackwardEvent', mouse_wheel_backward_event)

    for actor in actors:
        viewer.GetRenderer().AddActor(actor)

    viewer.Render()
    iren.Initialize()
    iren.Start()


def make_actor(polydata, color=None, width=None, opacity=None):
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    if color is not None:
        actor.GetProperty().SetColor(color)

    if width is not None:
        actor.GetProperty().SetLineWidth(width)
    else:
        actor.GetProperty().SetLineWidth(10.0)

    if opacity is not None:
        actor.GetProperty().SetOpacity(opacity)

    return actor


def show_polydata(polydata):
    actor = make_actor(polydata)

    ren = vtk.vtkRenderer()
    renwin = vtk.vtkRenderWindow()
    iren = vtk.vtkRenderWindowInteractor()
    renwin.AddRenderer(ren)
    iren.SetRenderWindow(renwin)

    ren.AddActor(actor)

    renwin.Render()
    iren.Start()


def show_interactive_image(image):
    lut = vtk.vtkLookupTable()
    lut.SetTableRange(0.0, 1.0)
    lut.SetHueRange(0.0, 0.0)
    lut.SetSaturationRange(0.0, 0.0)
    lut.SetValueRange(0.0, 1.0)

    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputData(image)
    # mapper.ScalarVisibilityOff()
    mapper.SetLookupTable(lut)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    ren = vtk.vtkRenderer()
    renwin = vtk.vtkRenderWindow()
    iren = vtk.vtkRenderWindowInteractor()
    renwin.AddRenderer(ren)
    iren.SetRenderWindow(renwin)

    ren.AddActor(actor)
    ren.SetBackground(1.0, 1.0, 1.0)

    renwin.Render()
    iren.Start()
