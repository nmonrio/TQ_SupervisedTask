def test_package_imports():
    import qml_bc

    assert isinstance(qml_bc.__version__, str)

