program test_fsz
    use iso_c_binding, only: c_size_t, c_int, c_int8_t, c_ptr, c_loc, c_null_ptr, &
                             c_float, c_associated
    use fsz
    implicit none

    interface
        function cudaMalloc(p, sz) bind(c, name='cudaMalloc') result(st)
            import :: c_ptr, c_size_t, c_int
            type(c_ptr), intent(out) :: p
            integer(c_size_t), value :: sz
            integer(c_int) :: st
        end function cudaMalloc
        function cudaFree(p) bind(c, name='cudaFree') result(st)
            import :: c_ptr, c_int
            type(c_ptr), value :: p
            integer(c_int) :: st
        end function cudaFree
        function cudaMemcpy(dst, src, sz, kind) bind(c, name='cudaMemcpy') result(st)
            import :: c_ptr, c_size_t, c_int
            type(c_ptr), value :: dst
            type(c_ptr), value :: src
            integer(c_size_t), value :: sz
            integer(c_int), value :: kind
            integer(c_int) :: st
        end function cudaMemcpy
        function cudaDeviceSynchronize() bind(c, name='cudaDeviceSynchronize') result(st)
            import :: c_int
            integer(c_int) :: st
        end function cudaDeviceSynchronize
    end interface

    integer(c_int), parameter :: H2D = 1_c_int, D2H = 2_c_int
    integer :: fails
    fails = 0

    call gate_r4()
    call gate_r8()
    call gate_small()
    call gate_rank3()
    call gate_error()
    call gate_device()

    if (fails == 0) then
        print '(a)', 'ALL PASS'
    else
        print '(a,i0)', 'FAILURES: ', fails
        error stop 1
    end if

contains

    subroutine gate_r4()
        integer(c_size_t), parameter :: n = 97_c_size_t * 113 * 131
        real(4), allocatable :: a(:), b(:)
        integer(c_int8_t), allocatable :: cmp(:)
        integer(c_size_t) :: cmp_size, i
        integer(c_int) :: ierr
        real(4), parameter :: eb = 1.0e-3
        allocate(a(n), b(n))
        do i = 1, n
            a(i) = sin(0.0017 * real(i)) + 0.05 * cos(0.13 * real(i))
        end do
        allocate(cmp(fsz_max_compressed_bytes(n)))
        call fsz_compress(a, n, eb, cmp, cmp_size, ierr)
        call check('compress-r4', ierr == FSZ_OK .and. cmp_size > 0)
        b = 0.0
        call fsz_decompress(b, cmp, cmp_size, n, eb, ierr)
        call check('roundtrip-r4', ierr == FSZ_OK .and. &
                   maxval(abs(a - b)) <= 1.01 * eb)
        print '(a,f0.3)', '  ratio r4: ', real(n) * 4.0 / real(cmp_size)
    end subroutine gate_r4

    subroutine gate_r8()
        integer(c_size_t), parameter :: n = 97_c_size_t * 113 * 131
        real(8), allocatable :: a(:), b(:)
        integer(c_int8_t), allocatable :: cmp(:)
        integer(c_size_t) :: cmp_size, i
        integer(c_int) :: ierr
        real(8), parameter :: eb = 1.0d-3
        allocate(a(n), b(n))
        do i = 1, n
            a(i) = sin(0.0017d0 * real(i, 8)) + 0.05d0 * cos(0.13d0 * real(i, 8))
        end do
        allocate(cmp(fsz_max_compressed_bytes(n)))
        call fsz_compress(a, n, eb, cmp, cmp_size, ierr)
        call check('compress-r8', ierr == FSZ_OK .and. cmp_size > 0)
        b = 0.0d0
        call fsz_decompress(b, cmp, cmp_size, n, eb, ierr)
        call check('roundtrip-r8', ierr == FSZ_OK .and. &
                   maxval(abs(a - b)) <= 1.01d0 * eb)
        print '(a,f0.3)', '  ratio r8: ', real(n) * 8.0 / real(cmp_size)
    end subroutine gate_r8

    subroutine gate_small()
        integer(c_size_t), parameter :: n = 4096_c_size_t
        real(4) :: a(n), b(n)
        integer(c_int8_t), allocatable :: cmp(:)
        integer(c_size_t) :: cmp_size, i
        integer(c_int) :: ierr
        real(4), parameter :: eb = 1.0e-3
        do i = 1, n
            a(i) = sin(0.05 * real(i)) + 0.2 * cos(0.31 * real(i))
        end do
        allocate(cmp(fsz_max_compressed_bytes(n)))
        call fsz_compress(a, n, eb, cmp, cmp_size, ierr)
        b = 0.0
        call fsz_decompress(b, cmp, cmp_size, n, eb, ierr)
        call check('small-array', ierr == FSZ_OK .and. &
                   cmp_size > fsz_num_tiles(n) * 8 .and. &
                   maxval(abs(a - b)) <= 1.01 * eb)
    end subroutine gate_small

    subroutine gate_rank3()
        integer, parameter :: nx = 32, ny = 32, nz = 32
        real(4) :: a(nx, ny, nz), b(nx, ny, nz)
        integer(c_int8_t), allocatable :: cmp(:)
        integer(c_size_t) :: n, cmp_size
        integer(c_int) :: ierr
        integer :: i, j, k
        real(4), parameter :: eb = 1.0e-3
        do k = 1, nz
            do j = 1, ny
                do i = 1, nx
                    a(i, j, k) = sin(0.1 * i) * cos(0.09 * j) + 0.02 * k
                end do
            end do
        end do
        n = int(nx, c_size_t) * ny * nz
        allocate(cmp(fsz_max_compressed_bytes(n)))
        call fsz_compress(a, n, eb, cmp, cmp_size, ierr)
        b = 0.0
        call fsz_decompress(b, cmp, cmp_size, n, eb, ierr)
        call check('rank-3', ierr == FSZ_OK .and. &
                   maxval(abs(a - b)) <= 1.01 * eb)
    end subroutine gate_rank3

    subroutine gate_error()
        real(4) :: a(16)
        integer(c_int8_t) :: cmp(4096)
        integer(c_size_t) :: cmp_size
        integer(c_int) :: ierr
        a = 1.0
        call fsz_compress(a, 16_c_size_t, 0.0, cmp, cmp_size, ierr)
        call check('error-status', ierr /= FSZ_OK .and. &
                   len(fsz_status_message(ierr)) > 0)
    end subroutine gate_error

    subroutine gate_device()
        integer(c_size_t), parameter :: n = 97_c_size_t * 113 * 131
        real(4), allocatable, target :: a(:), b(:)
        integer(c_int8_t), allocatable, target :: cmp_host(:)
        integer(c_size_t) :: host_size, cap, i
        integer(c_int) :: ierr, cst
        type(c_ptr) :: d_in, d_out, d_cmp, ws
        type(fsz_result) :: res
        real(4), parameter :: eb = 1.0e-3

        allocate(a(n), b(n))
        do i = 1, n
            a(i) = sin(0.0017 * real(i)) + 0.05 * cos(0.13 * real(i))
        end do

        cap = fsz_max_compressed_bytes(n)
        allocate(cmp_host(cap))
        call fsz_compress(a, n, eb, cmp_host, host_size, ierr)
        call check('device-setup-host', ierr == FSZ_OK)

        cst = cudaMalloc(d_in,  n * 4_c_size_t)
        if (cst /= 0) then
            call check('device-alloc', .false.)
            return
        end if
        cst = cudaMalloc(d_out, n * 4_c_size_t)
        cst = cudaMalloc(d_cmp, cap)
        cst = cudaMemcpy(d_in, c_loc(a), n * 4_c_size_t, H2D)
        call check('device-h2d', cst == 0)

        call fsz_workspace_create(ws, n, ierr)
        call check('workspace-create', ierr == FSZ_OK .and. c_associated(ws) .and. &
                   fsz_workspace_capacity(ws) == n)

        call fsz_compress_device(d_in, d_cmp, n, eb, ws, res, ierr)
        call check('compress-device', ierr == FSZ_OK .and. res%cmp_size == host_size)

        call fsz_decompress_device(d_out, d_cmp, n, eb, res, ws, ierr)
        cst = cudaDeviceSynchronize()
        b = 0.0
        cst = cudaMemcpy(c_loc(b), d_out, n * 4_c_size_t, D2H)
        call check('decompress-device', ierr == FSZ_OK .and. cst == 0 .and. &
                   maxval(abs(real(a, 8) - real(b, 8))) <= 1.01d0 * real(eb, 8))

        block
            type(fsz_result) :: res2
            call fsz_compress_device(d_in, d_cmp, n, eb, c_null_ptr, res2, ierr)
            call check('compress-device-nullws', ierr == FSZ_OK .and. &
                       res2%cmp_size == host_size)
        end block

        print '(a,i0,a)', '  device stream ', res%cmp_size, ' bytes, matches the host path'

        call fsz_workspace_destroy(ws)
        cst = cudaFree(d_in); cst = cudaFree(d_out); cst = cudaFree(d_cmp)
    end subroutine gate_device

    subroutine check(name, ok)
        character(len=*), intent(in) :: name
        logical, intent(in) :: ok
        if (ok) then
            print '(a,a)', '[PASS] ', name
        else
            print '(a,a)', '[FAIL] ', name
            fails = fails + 1
        end if
    end subroutine check

end program test_fsz
