// vtk.js ships partial TS types; this wildcard keeps the build green while the VIEW
// layer uses loose vtk types. The MODEL layer stays fully, strictly typed.
declare module '@kitware/vtk.js/*';
