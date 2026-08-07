program example_fsz
    use iso_c_binding, only: c_size_t, c_int, c_int8_t
    use fsz
    implicit none

    integer, parameter :: nx = 64, ny = 64, nz = 64
    real(4) :: field(nx, ny, nz), recon(nx, ny, nz)
    integer(c_int8_t), allocatable :: cmp(:)
    integer(c_size_t) :: n, cmp_size
    integer(c_int) :: ierr
    real(4) :: eb, max_err
    integer :: i, j, k

    do k = 1, nz
        do j = 1, ny
            do i = 1, nx
                field(i, j, k) = sin(0.05 * i) * cos(0.04 * j) &
                               + 0.1 * sin(0.11 * k)
            end do
        end do
    end do

    n  = int(nx, c_size_t) * ny * nz
    eb = 1.0e-3
    allocate(cmp(fsz_max_compressed_bytes(n)))

    call fsz_compress(field, n, eb, cmp, cmp_size, ierr)
    if (ierr /= FSZ_OK) then
        print '(a)', 'compression failed: ' // fsz_status_message(ierr)
        error stop 1
    end if

    call fsz_decompress(recon, cmp, cmp_size, n, eb, ierr)
    if (ierr /= FSZ_OK) then
        print '(a)', 'decompression failed: ' // fsz_status_message(ierr)
        error stop 1
    end if

    max_err = maxval(abs(field - recon))
    print '(a,a)',        'FSZ       : ', fsz_version()
    print '(a,f0.3)',     'ratio     : ', real(n) * 4.0 / real(cmp_size)
    print '(a,es10.3,a,es10.3)', 'max error : ', max_err, '  bound ', eb
    if (max_err <= 1.01 * eb) then
        print '(a)', 'PASS'
    else
        print '(a)', 'FAIL'
        error stop 1
    end if
end program example_fsz
