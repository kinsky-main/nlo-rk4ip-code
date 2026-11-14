## nlolib @VERSION@ — one MATLAB toolbox for every platform

`nlolib.mltbx` carries the Windows and Linux binaries side by side under
`lib/win64` and `lib/glnxa64`, each with its own `loadlibrary` prototype, thunk
and progress MEX. MATLAB loads the folder matching `computer('arch')`, so the
same download works on both platforms.

### Install

From the [File Exchange entry](https://www.mathworks.com/matlabcentral/fileexchange/185226-nlolib),
or **Home → Add-Ons → Get Add-Ons** inside MATLAB and search for `nlolib`.

To install straight from this release instead:

```matlab
f = websave([tempname '.mltbx'], 'https://github.com/kinsky-main/nlo-rk4ip-code/releases/latest/download/nlolib.mltbx');
matlab.addons.toolbox.installToolbox(f);
```

That URL always resolves to the current release. To pin this one, replace
`latest/download` with `download/@VERSION@`.

Requires MATLAB R2023b or later on 64-bit Windows or 64-bit Linux. Nothing else
is needed: no C compiler, no Visual C++ redistributable and no Vulkan SDK — the
prototype and thunk are prebuilt and the library links its runtime statically. A
GPU driver carrying the Vulkan loader enables the GPU backend; without one the
CPU backend runs unchanged.

Verify, and add the examples for the current session:

```matlab
api = nlolib.NLolib();
nlolib_setup();
```

On Linux, install from the MATLAB desktop or with
`matlab -nodisplay -r "matlab.addons.toolbox.installToolbox('nlolib.mltbx'); exit"`.
A `matlab -batch` install extracts the files but its add-on registration does not
persist, so the next session finds nothing.

### Upgrading

Every bundle shares one fixed toolbox GUID, so installing a newer
`nlolib.mltbx` replaces the installed one in place. Installs made before 1.13.0,
and from the first 1.13.0 build, carry a different identifier and must be
uninstalled by hand — the user guide has a snippet.

Full documentation:
[`docs/matlab_user_guide.md`](https://github.com/kinsky-main/nlo-rk4ip-code/blob/main/docs/matlab_user_guide.md)
