# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import base64
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image
from pypdf import PdfWriter
from pypdf.generic import RectangleObject

from haystack.components.converters.image.image_utils import _convert_pdf_to_images
from haystack.components.converters.image.pdf_to_image import PDFToImageContent
from haystack.core.serialization import component_from_dict, component_to_dict
from haystack.dataclasses import ByteStream


class TestPDFToImageContent:
    @pytest.mark.parametrize(
        "media_box,crop_box",
        [
            ((10000, 10000, 10072, 10144), None),
            ((-10072, -10144, -10000, -10000), None),
            ((0, 0, 10000, 10000), (1000, 2000, 1072, 2144)),
        ],
        ids=["positive-origin", "negative-origin", "cropped-page"],
    )
    def test_run_preserves_resolution_for_offset_or_cropped_pages(
        self, media_box: tuple[int, int, int, int], crop_box: tuple[int, int, int, int] | None
    ) -> None:
        writer = PdfWriter()
        page = writer.add_blank_page(width=72, height=144)
        page.mediabox = RectangleObject(media_box)
        if crop_box is not None:
            page.cropbox = RectangleObject(crop_box)
        stream = BytesIO()
        writer.write(stream)

        result = PDFToImageContent().run(sources=[ByteStream(data=stream.getvalue(), mime_type="application/pdf")])

        assert len(result["image_contents"]) == 1
        image_content = result["image_contents"][0]
        with Image.open(BytesIO(base64.b64decode(image_content.base64_image))) as image:
            # A 72-by-144-point visible page is one by two inches, rendered at 300 DPI.
            assert image.size == (300, 600)

    def test_run_limits_pixels_for_page_ending_at_origin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        writer = PdfWriter()
        page = writer.add_blank_page(width=72, height=72)
        page.mediabox = RectangleObject((-72, -72, 0, 0))
        stream = BytesIO()
        writer.write(stream)

        with monkeypatch.context() as patcher:
            patcher.setattr(Image, "MAX_IMAGE_PIXELS", 10000)
            result = PDFToImageContent().run(sources=[ByteStream(data=stream.getvalue(), mime_type="application/pdf")])

        assert len(result["image_contents"]) == 1
        with Image.open(BytesIO(base64.b64decode(result["image_contents"][0].base64_image))) as image:
            assert image.width * image.height <= 10000

    def test_to_dict(self) -> None:
        converter = PDFToImageContent()
        assert component_to_dict(converter, "converter") == {
            "init_parameters": {"detail": None, "size": None, "page_range": None},
            "type": "haystack.components.converters.image.pdf_to_image.PDFToImageContent",
        }

    def test_to_dict_not_defaults(self) -> None:
        converter = PDFToImageContent(detail="low", size=(128, 128), page_range=[1])
        assert component_to_dict(converter, "converter") == {
            "init_parameters": {"detail": "low", "size": (128, 128), "page_range": [1]},
            "type": "haystack.components.converters.image.pdf_to_image.PDFToImageContent",
        }

    def test_from_dict(self) -> None:
        data = {
            "init_parameters": {"detail": "auto", "size": None, "page_range": [1]},
            "type": "haystack.components.converters.image.pdf_to_image.PDFToImageContent",
        }
        converter = component_from_dict(PDFToImageContent, data, "name")
        assert component_to_dict(converter, "converter") == data

    def test_run_with_valid_source(self) -> None:
        file_path = "./test/test_files/pdf/sample_pdf_1.pdf"
        mime_type = "application/pdf"
        converter = PDFToImageContent()
        results = converter.run(sources=[file_path])

        byte_stream = ByteStream.from_file_path(Path(file_path), mime_type=mime_type, meta={"file_path": file_path})
        assert len(results["image_contents"]) == 4
        assert results["image_contents"][0].base64_image is not None
        assert (
            results["image_contents"][0].base64_image
            == _convert_pdf_to_images(bytestream=byte_stream, size=None, page_range=[1], return_base64=True)[0][1]
        )
        assert results["image_contents"][0].mime_type == "image/jpeg"
        assert results["image_contents"][0].detail is None
        assert results["image_contents"][0].meta["file_path"] == str(Path(file_path))
        assert results["image_contents"][0].meta["page_number"] == 1
        assert results["image_contents"][1].meta["page_number"] == 2
        assert results["image_contents"][2].meta["page_number"] == 3
        assert results["image_contents"][3].meta["page_number"] == 4

    def test_run_with_no_sources(self) -> None:
        converter = PDFToImageContent()
        results = converter.run(sources=[])
        assert len(results["image_contents"]) == 0
        assert results == {"image_contents": []}

    def test_run_with_invalid_source_type(self, caplog: pytest.LogCaptureFixture) -> None:
        converter = PDFToImageContent()
        # The invalid source type is what this test exercises.
        converter.run(sources=[123])  # type: ignore[list-item]
        assert "Could not read" in caplog.text

    def test_run_with_non_existent_file(self, caplog: pytest.LogCaptureFixture) -> None:
        converter = PDFToImageContent()
        converter.run(sources=["./non_existent_file.png"])
        assert "Could not read" in caplog.text
        assert "No such file or directory:" in caplog.text

    def test_run_with_bytestream_sources(self) -> None:
        file_path = "./test/test_files/pdf/sample_pdf_1.pdf"
        mime_type = "application/pdf"
        byte_stream = ByteStream.from_file_path(Path(file_path), mime_type=mime_type, meta={"file_path": file_path})

        # Initialize the converter
        converter = PDFToImageContent()

        # Run the converter with the ByteStream
        results = converter.run(sources=[byte_stream])

        # Assertions
        assert len(results["image_contents"]) == 4
        assert results["image_contents"][0].base64_image is not None
        assert (
            results["image_contents"][0].base64_image
            == _convert_pdf_to_images(bytestream=byte_stream, size=None, page_range=[1], return_base64=True)[0][1]
        )
        assert results["image_contents"][0].mime_type == "image/jpeg"
        assert results["image_contents"][0].detail is None
        assert results["image_contents"][0].meta["file_path"] == file_path
        assert results["image_contents"][0].meta["page_number"] == 1

    def test_run_with_empty_bytestream(self) -> None:
        # Create an empty ByteStream object
        byte_stream = ByteStream(data=b"", meta={"file_path": "empty_file.pdf"})

        # Initialize the converter
        converter = PDFToImageContent()

        # Run the converter with the empty ByteStream
        results = converter.run(sources=[byte_stream])

        assert results["image_contents"] == []
